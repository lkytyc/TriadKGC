### Perform ontology evolution task based on preprocessed KG data ###
import asyncio
import json
from asyncio import as_completed
from typing import List, Any
import csv
from tqdm import tqdm

from Triad_KGC.code.agents.agent_factory import AgentFactory
from Triad_KGC.code.types.models import Node, Triples
from Triad_KGC.code.types.kg_operator import KGOperator
from Triad_KGC.code.utils.res_parse.evolution import parse_kg_from_string

# Create intelligent agents for triple completion, and background generation
triples_completion_agent = AgentFactory.create_agent("evolution", "TriplesCompletionAgent")
background_generation_agent = AgentFactory.create_agent("evolution", "BackgroundGenerationAgent")


class OntologyEvolutionController:
    def __init__(self, kg_operator, schema_kg_operator):
        """
        Initialize ontology evolution controller

        Args:
        - kg_operator: KGOperator instance for instance-level KG operations
        - schema_kg_operator: KGOperator instance for schema-level KG operations
        """
        self.kg_operator: KGOperator = kg_operator
        self.schema_kg_operator: KGOperator = schema_kg_operator
        self.semaphore = asyncio.Semaphore(50)  # Limit concurrent tasks to avoid overload

    def process_dataset_entity_triples_and_descriptions(self,
                                                        dateset_kg: list,
                                                        dateset_entities_desc: list,
                                                        entity_name: str,
                                                        mask_entity_name: str):
        """
        Given triples and entity descriptions, return related triples and descriptions for a target entity.

        Args:
        - dateset_kg: List of triples with fields 'head', 'relation', 'tail'
        - dateset_entities_desc: List of dicts with 'entity', 'entity_desc'
        - entity_name: Target entity name
        - mask_entity_name: Masked entity name

        Returns:
        - matched_triples: List of matching triples
        - matched_descriptions: Dict mapping entity names to description text
        """
        matched_triples = []
        for t in dateset_kg:
            head = t.get('head')
            tail = t.get('tail')
            rel = t.get('relation')
            if (head == entity_name or tail == entity_name) and mask_entity_name not in (head, tail):
                matched_triples.append((head, rel, tail))

        # Build entity description mapping
        desc_mapping = {e['entity']: e.get('entity_desc', '') for e in dateset_entities_desc}

        # Collect matched entities
        matched_entities = {head for head, _, _ in matched_triples} | {tail for _, _, tail in matched_triples}

        # Get descriptions of matched entities
        matched_descriptions = {ent: desc_mapping.get(ent, '') for ent in matched_entities}

        return matched_triples, matched_descriptions

    async def find_all_type_triples(self) -> list[Triples]:
        """
        Get all type-level triples (i.e., schema-level KG edges)

        Returns:
        - type_triples: List of type-level triples
        """
        type_triples = []
        for edge in self.schema_kg_operator.kg.edges:
            source_entity = await self.schema_kg_operator.find_node_by_id(edge.source)
            target_entity = await self.schema_kg_operator.find_node_by_id(edge.target)
            type_triples.append(Triples(source_entity=source_entity, target_entity=target_entity, edge=edge))
        return type_triples

    async def _find_user_expected_complete_triples(self, user_expected_type_triples: list[Triples],
                                                   dateset_kg: list,
                                                   dateset_entities_desc: list) -> list[tuple]:
        """
        From type-level triples, find instance-level triples the user expects to complete.
        Construct gold and masked triples and gather background knowledge for the known entity.

        Args:
        - user_expected_type_triples: List of type-level triples
        - dateset_kg: List of instance-level triples
        - dateset_entities_desc: Entity description data

        Returns:
        - List of tuples containing gold triple, masked triple, and mask position
        """
        seen = set()

        async def process_type_triple(type_triple):
            results = []
            source_type_entity = type_triple.source_entity
            target_type_entity = type_triple.target_entity
            edge_type = type_triple.edge

            if edge_type.attributes.get("source_user_add", '') != "true":
                # Mask the target entity (tail)
                source_instances = await self.kg_operator.find_nodes_by_type(type_=source_type_entity.name)
                for source_node in source_instances:
                    edges = await self.kg_operator.find_edges_by_source(source_id=source_node.id)
                    for edge in edges:
                        if edge.type == edge_type.name:
                            target_node = await self.kg_operator.find_node_by_id(edge.target)
                            if target_node.type == target_type_entity.name:
                                key = (source_node.id, edge.name, target_node.id)
                                if key in seen:
                                    continue
                                seen.add(key)
                                gold_triple = Triples(source_entity=source_node, target_entity=target_node, edge=edge)
                                background_triples, entities_description = self.process_dataset_entity_triples_and_descriptions(
                                    dateset_kg=dateset_kg, entity_name=source_node.name,
                                    mask_entity_name=target_node.name,
                                    dateset_entities_desc=dateset_entities_desc)
                                if background_triples or entities_description:
                                    background_info = await self._generate_background(
                                        entity=source_node,
                                        type_definition=source_type_entity.attributes.get("definition", ""),
                                        background_triples=background_triples,
                                        entities_description=entities_description)
                                    edge.attributes["background_information"] = background_info

                                masked_target = Node(id=target_node.id, type=target_node.type, name=" ", attributes=target_node.attributes)
                                masked_triple = Triples(source_entity=source_node, target_entity=masked_target, edge=edge)
                                results.append((gold_triple, masked_triple, "tail"))

            else:
                # Mask the source entity (head)
                target_instances = await self.kg_operator.find_nodes_by_type(type_=target_type_entity.name)
                for target_node in target_instances:
                    edges = await self.kg_operator.find_edges_by_target(target_id=target_node.id)
                    for edge in edges:
                        if edge.type == edge_type.name:
                            source_node = await self.kg_operator.find_node_by_id(edge.source)
                            if source_node.type == source_type_entity.name:
                                key = (source_node.id, edge.name, target_node.id)
                                if key in seen:
                                    continue
                                seen.add(key)
                                gold_triple = Triples(source_entity=source_node, target_entity=target_node, edge=edge)
                                background_triples, entities_description = self.process_dataset_entity_triples_and_descriptions(
                                    dateset_kg=dateset_kg, entity_name=target_node.name, mask_entity_name=source_node.name,
                                    dateset_entities_desc=dateset_entities_desc)
                                if background_triples or entities_description:
                                    background_info = await self._generate_background(
                                        entity=target_node,
                                        type_definition=target_type_entity.attributes.get("definition", ""),
                                        background_triples=background_triples,
                                        entities_description=entities_description)
                                    edge.attributes["background_information"] = background_info

                                masked_source = Node(id=source_node.id, type=source_node.type, name=" ", attributes=source_node.attributes)
                                masked_triple = Triples(source_entity=masked_source, target_entity=target_node, edge=edge)
                                results.append((gold_triple, masked_triple, "head"))

            return results

        all_results = await asyncio.gather(*(process_type_triple(tt) for tt in user_expected_type_triples))
        return [item for sublist in all_results for item in sublist]

    @staticmethod
    async def _generate_background(background_triples: list, entity: Node, type_definition: str, entities_description: dict) -> Any | None:
        """
        Use the background generation agent to generate background information for the known entity.

        Args:
        - background_triples: Triples involving the known entity (excluding gold triple)
        - entity: The known entity node
        - type_definition: Definition of the known entity’s type
        - entities_description: Entity description dictionary

        Returns:
        - Background text information about the known entity
        """
        try:
            background_triples_str = json.dumps([f"({h}, {r}, {t})" for h, r, t in background_triples], ensure_ascii=False)
            return await background_generation_agent.run(
                background_triples=background_triples_str,
                core_entity_information=json.dumps({
                    "name": entity.name,
                    "type": entity.type,
                    "type_definition": type_definition
                }),
                background_entities_description=json.dumps(entities_description)
            )
        except Exception as e:
            print(f"Error in generate background information: {e}")

    @staticmethod
    async def _complete(user_expected_complete_triple: Triples,
                        source_definition: str,
                        target_definition: str,
                        edge_definition) -> List[Triples] | None:
        """
        Use agent to complete a masked triple.

        Args:
        - user_expected_complete_triple: Masked triple (with blank entity)
        - source_definition: Source type definition
        - target_definition: Target type definition
        - edge_definition: Relation definition

        Returns:
        - List of completed triples
        """
        try:
            async def do_completion():
                return await triples_completion_agent.run(
                    incomplete_triple=f"({user_expected_complete_triple.source_entity.name}, {user_expected_complete_triple.edge.name}, {user_expected_complete_triple.target_entity.name})",
                    description_of_known_entity=user_expected_complete_triple.source_entity.attributes['desc']
                    if user_expected_complete_triple.source_entity.name
                    else user_expected_complete_triple.target_entity.attributes['desc'],
                    completed_entity_type=user_expected_complete_triple.target_entity.type
                    if user_expected_complete_triple.source_entity.name
                    else user_expected_complete_triple.source_entity.type,
                    source_definition=source_definition,
                    target_definition=target_definition,
                    edge_definition=edge_definition,
                    background_context=user_expected_complete_triple.edge.attributes.get("background_information", "No context")
                )

            completed_kg = parse_kg_from_string(await do_completion(), masked_triples=user_expected_complete_triple)

            def not_match(triple):
                if user_expected_complete_triple.source_entity.name:
                    return triple.target_entity.name.strip().lower() != user_expected_complete_triple.target_entity.name.strip().lower()
                else:
                    return triple.source_entity.name.strip().lower() != user_expected_complete_triple.source_entity.name.strip().lower()

            if all(not_match(triple) for triple in completed_kg):
                completed_kg = parse_kg_from_string(await do_completion(), masked_triples=user_expected_complete_triple)

            return completed_kg

        except Exception as e:
            print(f"Error in completing triple: {e}")
            return None

    async def complete(self, user_expected_complete_triple: Triples,
                       source_definition: str,
                       target_definition: str,
                       edge_definition) -> List[Triples]:
        """
        Complete a masked triple with semaphore control for concurrency.

        Args:
        - user_expected_complete_triple: Masked triple
        - source_definition, target_definition, edge_definition: Definitions related to the triple

        Returns:
        - List of completed triples
        """
        async with self.semaphore:
            return await self._complete(user_expected_complete_triple, source_definition, target_definition, edge_definition)

    async def run_user_expected_triples_completion(self, gold_and_masked_triples: List[tuple]) -> List[tuple]:
        """
        Complete all masked triples.

        Args:
        - gold_and_masked_triples: List of (gold_triple, masked_triple, mask_position)

        Returns:
        - List of (gold_triple, completed triple string, mask_position)
        """
        tasks = []
        for gold_triple, masked_triple, mask_position in gold_and_masked_triples:
            source_type = await self.schema_kg_operator.find_node_by_name(gold_triple.source_entity.type)
            target_type = await self.schema_kg_operator.find_node_by_name(gold_triple.target_entity.type)
            edge_type = await self.schema_kg_operator.find_edge_by_source_target(source_id=source_type.id, target_id=target_type.id)

            if source_type and target_type and edge_type:
                tasks.append(self.complete(masked_triple,
                                           source_definition=source_type.attributes.get("definition"),
                                           target_definition=target_type.attributes.get("definition"),
                                           edge_definition=edge_type.attributes.get("definition")))

        completed_results = []
        with tqdm(total=len(tasks), desc="Completing User Expected Triples") as pbar:
            completed_results = await asyncio.gather(*tasks)
            pbar.update(len(tasks))

        gold_and_completed_triples = []
        for (gold_triple, masked_triple, mask_position), completed_list in zip(gold_and_masked_triples, completed_results):
            completed_triples_str = ", ".join(
                f"({t.source_entity.name}, {t.edge.name}, {t.target_entity.name})"
                for t in completed_list
            ) if completed_list else ""
            gold_and_completed_triples.append((gold_triple, completed_triples_str, mask_position))

        return gold_and_completed_triples

    async def execute_evolution(self, dateset_kg: list, dateset_entities_desc: list) -> List[tuple]:
        """
        Execute the full KG ontology evolution and completion process.

        Args:
        - dateset_kg: List of triples
        - dateset_entities_desc: Entity descriptions

        Returns:
        - List of completed triples
        """
        all_type_triples = await self.find_all_type_triples()
        user_expected_complete_triples = await self._find_user_expected_complete_triples(all_type_triples, dateset_kg, dateset_entities_desc)
        all_complete_triples = await self.run_user_expected_triples_completion(user_expected_complete_triples)
        return all_complete_triples


if __name__ == "__main__":
    # Execute KG completion process and save results to CSV
    with open("../../input/test_kg.json", 'r', encoding='utf-8') as f:
        test_kg_json_str = f.read()
    with open("../../input/test_schema.json", 'r', encoding='utf-8') as f:
        schema_json_str = f.read()

    dataset = "WN18RR"   # FB15k237
    with open(f"../../../data/{dataset}/dataset.json", "r", encoding="utf-8") as f:
        dateset_kg = json.load(f)
    with open(f"../../../data/{dataset}/entities.json", "r", encoding="utf-8") as f:
        dateset_entities_desc = json.load(f)

    kg_operator = asyncio.run(KGOperator.from_json(json_str=test_kg_json_str))
    schema_kg_operator = asyncio.run(KGOperator.from_json(json_str=schema_json_str))
    ontology_evolution_controller = OntologyEvolutionController(kg_operator, schema_kg_operator)

    all_complete_triples = asyncio.run(
        ontology_evolution_controller.execute_evolution(dateset_kg, dateset_entities_desc)
    )

    with open("../../output/completed_triples.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Gold Triple", "Completed Triples", "Mask Position", "Type Triple"])
        for gold_triple, completed_triples_str, mask_position in all_complete_triples:
            gold_str = f"({gold_triple.source_entity.name}, {gold_triple.edge.name}, {gold_triple.target_entity.name})"
            type_str = f"({gold_triple.source_entity.type}, {gold_triple.edge.type}, {gold_triple.target_entity.type})"
            if gold_str and completed_triples_str:
                writer.writerow([gold_str, completed_triples_str, mask_position, type_str])
