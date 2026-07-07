### Receive the evolved partial KG and verify each triple with triple verification ###
import asyncio
import csv
import re
from typing import List

from tqdm import tqdm

from Triad_KGC.code.types.models import Node, Edge, KG, Triples
from Triad_KGC.code.kg_verification.web_search.main import get_web_context_blocks
from Triad_KGC.code.kg_verification.retrieve.related_retrieve import RelatedRetriever
from Triad_KGC.code.agents.agent_factory import AgentFactory

# Initialize agents
claim_verification_agent = AgentFactory.create_agent(agent_type="verification", agent="ClaimVerificationAgent")
judgement_verification_agent = AgentFactory.create_agent(agent_type="verification", agent="JudgementVerificationAgent")
selection_verification_agent = AgentFactory.create_agent(agent_type="verification", agent="SelectionVerificationAgent")
interference_generation_agent = AgentFactory.create_agent(agent_type="verification", agent="InterferenceGenerationAgent")
domain_knowledge_generation_agent = AgentFactory.create_agent(agent_type="verification", agent="DomainKnowledgeGenerationAgent")


class KGVerification:
    """
    Verifies the validity and reliability of the evolved partial KG.
    """

    def __init__(self):
        self.related_retriever = RelatedRetriever(top_k=3)
        self.semaphore = asyncio.Semaphore(20)

    async def limited_triple_verification(self, unvalidated_triples: Triples) -> bool:
        """
        Triple verification method with concurrency control.
        :param unvalidated_triples: Triple to verify
        :return: bool
        """
        async with self.semaphore:
            return await self.triples_verification(unvalidated_triples=unvalidated_triples)

    async def run_triple_verification(self, unvalidated_triples: Triples) -> bool:
        """
        Triple verification
        :param unvalidated_triples:
        :return:
        """
        domain_knowledge = await domain_knowledge_generation_agent.run(
            entity=unvalidated_triples.source_entity.name,
        )
        domain_knowledge += await domain_knowledge_generation_agent.run(
            entity=unvalidated_triples.target_entity.name,
        )

        # Selection verification requires generation of distractors, wrapped as an internal async task
        async def selection_verification_task(triple: Triples):
            # Generate distractors
            incomplete_triples = f"({triple.source_entity.name},  , {triple.target_entity.name})"
            interference = await interference_generation_agent.run(
                incomplete_triples=incomplete_triples
            )
            # Add the original relation to choices
            choices = f"{interference}, {triple.edge.name}"
            # Run selection verification
            selection_result = await selection_verification_agent.run(
                domain_knowledge=domain_knowledge,
                incomplete_triples=incomplete_triples,
                choices=choices,
            )
            if triple.edge.name.strip() in selection_result.strip():
                return True
            else:
                return False

        # Run claim, judgement, and selection verifications concurrently
        tasks = [
            claim_verification_agent.run(
                domain_knowledge=domain_knowledge,
                unvalidated_triples=f"({unvalidated_triples.source_entity.name}, {unvalidated_triples.edge.name}, {unvalidated_triples.target_entity.name})",
            ),
            judgement_verification_agent.run(
                domain_knowledge=domain_knowledge,
                source_entity=unvalidated_triples.source_entity.name,
                relationship=unvalidated_triples.edge.name,
                target_entity=unvalidated_triples.target_entity.name,
            ),
            selection_verification_task(unvalidated_triples)
        ]
        # Execute all tasks
        results = await asyncio.gather(*tasks)
        # Parse results
        claim_result = results[0].strip()
        judgement_result = results[1].strip()
        selection_result = results[2]
        # Final decision
        if "correct" in claim_result.lower() and "yes" in judgement_result.lower() and selection_result:
            return True
        else:
            return False

    async def run_web_search_verification(self, unvalidated_triples: Triples) -> bool:
        """
        Web search-based verification
        :param unvalidated_triples:
        :return:
        """
        # First, use web search to obtain context blocks
        web_context_blocks = await get_web_context_blocks(
            source_entity=unvalidated_triples.source_entity.name,
            relation=unvalidated_triples.edge.name,
            target_entity=unvalidated_triples.target_entity.name,
        )
        # Retrieve relevant blocks
        related_blocks = await self.related_retriever.retrieve_related_blocks(
            text_blocks=web_context_blocks,
            source_entity=unvalidated_triples.source_entity.name,
            relation=unvalidated_triples.edge.name,
            target_entity=unvalidated_triples.target_entity.name,
        )
        # If blocks found, return True
        if related_blocks:
            return True

    async def triples_verification(self, unvalidated_triples: Triples) -> bool:
        """
        Web search and triple expert verification
        :return: bool
        """
        # Try web search first
        web_search_result = await self.run_web_search_verification(unvalidated_triples=unvalidated_triples)
        if web_search_result:
            return True
        else:
            # Fallback to triple verification
            return await self.run_triple_verification(unvalidated_triples=unvalidated_triples)

    async def verify_kg(self, unvalidated_triples: List[List[Triples]]) -> List[List[Triples]]:
        """
        Verify KG's validity and reliability (async + tqdm progress)
        :param unvalidated_triples: 2D list of triples to verify
        :return: Verified triples in same 2D structure
        """
        verified_triples_count = 0
        tasks = []
        index_map = []  # (i, j) tracks the location of each triple in the original 2D list

        for i, triple_group in enumerate(unvalidated_triples):
            for j, triple in enumerate(triple_group):
                tasks.append(triple)
                index_map.append((i, j))

        if not tasks:
            return [[] for _ in unvalidated_triples]

        async def verify_task(triple: Triples) -> bool:
            return await self.limited_triple_verification(triple)

        verified_results = []
        progress_bar = tqdm(total=len(tasks), desc="Verifying Triples", ncols=80)

        async def verify_task_with_progress(triple: Triples) -> bool:
            result = await verify_task(triple)
            progress_bar.update(1)
            return result

        # Run all verification tasks concurrently and maintain order
        verified_results = await asyncio.gather(*(verify_task_with_progress(triple) for triple in tasks))
        progress_bar.close()

        # Count how many passed
        verified_triples_count = sum(1 for passed in verified_results if passed)

        # Reconstruct 2D structure
        verified_triples_2d: List[List[Triples]] = [[] for _ in range(len(unvalidated_triples))]
        for (i, _), passed, triple in zip(index_map, verified_results, tasks):
            if passed:
                verified_triples_2d[i].append(triple)

        survival_rate = verified_triples_count / len(tasks)
        print(f"[VerificationAgent]Survival Rate: {survival_rate:.2%}")
        if survival_rate < 0.1:
            import warnings
            warnings.warn(
                "Survival rate is low, consider stopping the evolution process.",
                UserWarning,
            )

        return verified_triples_2d


def parse_node(name: str, type_: str) -> Node:
    """
    Create a node. `name` is also the ID. `type_` is from the type triple.
    """
    return Node(id=name, type=type_, name=name)


def parse_edge(relation: str, source_id: str, target_id: str) -> Edge:
    """
    Create an edge. Both `type` and `name` are set to the relation string.
    """
    return Edge(source=source_id, target=target_id, type=relation, name=relation)


def parse_type_triple(triple_str: str) -> (str, str, str):
    """
    Parse a type triple string like "(source_type, relation, target_type)" and return source_type, relation, target_type
    """
    triple_str = triple_str.strip()
    if triple_str.startswith("(") and triple_str.endswith(")"):
        triple_str = triple_str[1:-1].strip()
    parts = re.split(r'\s*,\s*', triple_str, maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Type triple format error: {triple_str}")
    return parts[0], parts[1], parts[2]


def parse_instance_triple_str(triple_str: str, source_type: str, target_type: str) -> Triples:
    """
    Parse a single instance triple string "(source_name, relation, target_name)"
    Apply node types from the type triple
    """
    triple_str = triple_str.strip()
    if triple_str.startswith("(") and triple_str.endswith(")"):
        triple_str = triple_str[1:-1].strip()
    parts = re.split(r'\s*,\s*', triple_str, maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Instance triple format error: {triple_str}")
    source_name, relation, target_name = parts
    source_node = parse_node(source_name, source_type)
    target_node = parse_node(target_name, target_type)
    edge = parse_edge(relation, source_node.id, target_node.id)

    return Triples(source_entity=source_node, target_entity=target_node, edge=edge)


def parse_instance_triples_str(instances_str: str, source_type: str, target_type: str) -> List[Triples]:
    """
    Parse multiple instance triples string: "(a, r, b), (c, r, d), ..."
    """
    if not instances_str.strip():
        return []
    split_marker = "|"
    tmp_str = instances_str.replace("),", ")" + split_marker)
    parts = [p.strip() for p in tmp_str.split(split_marker) if p.strip()]
    triples = []
    for part in parts:
        triples.append(parse_instance_triple_str(part, source_type, target_type))
    return triples


def read_instance_triples_from_csv(file_path: str) -> List[List[Triples]]:
    """
    Read CSV and return a 2D list of instance triples. Node types are inferred from the type triple on the same row.
    """
    all_instance_triples = []

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            type_triple_str = row.get("Type Triple", "")
            instance_triples_str = row.get("Completed Triples", "")
            if not type_triple_str.strip():
                raise ValueError("Missing Type Triple in CSV row")
            source_type, relation, target_type = parse_type_triple(type_triple_str)
            instance_triples = parse_instance_triples_str(instance_triples_str, source_type, target_type)
            all_instance_triples.append(instance_triples)
    return all_instance_triples


if __name__ == "__main__":
    # Read triples from CSV
    completed_triples_path = "../../output/completed_triples.csv"
    verified_triples_path = "../../output/verified_triples.csv"
    all_triples = read_instance_triples_from_csv(file_path=completed_triples_path)
    kg_verification = KGVerification()
    # Verify KG
    verified_triples = asyncio.run(kg_verification.verify_kg(unvalidated_triples=all_triples))

    # Step 1: Build mapping from completed triples to gold triples
    completed_to_gold_mapping = {}
    gold_to_position_mapping = {}
    with open(completed_triples_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold_triple = row["Gold Triple"]
            completed_triples = row["Completed Triples"]
            mask_position = row["Mask Position"]
            type_triple = row["Type Triple"]

            completed_to_gold_mapping[completed_triples] = gold_triple
            gold_to_position_mapping[gold_triple] = mask_position

    # Step 2: Process verified triples
    csv_result = {}
    for complete_triples in verified_triples:
        if not complete_triples:
            continue
        type_triple = f"({complete_triples[0].source_entity.type}, {complete_triples[0].edge.type}, {complete_triples[0].target_entity.type})"
        instance_triples = ""
        for triple in complete_triples:
            instance_triples += f"({triple.source_entity.name}, {triple.edge.name}, {triple.target_entity.name}), "
        instance_triples = instance_triples[:-2]  # Remove trailing comma and space
        csv_result[instance_triples] = type_triple

    # Step 3: Write final CSV with gold triples
    with open(verified_triples_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Gold Triple", "Completed Triples", "Mask Position", "Type Triple"])
        for instance_triples, type_triple in csv_result.items():
            if type_triple and instance_triples:
                gold_triple = ""
                mask_position = ""
                for completed_key, gold_value in completed_to_gold_mapping.items():
                    if instance_triples in completed_key:
                        gold_triple = gold_value
                        mask_position = gold_to_position_mapping.get(gold_triple, "")
                        break
                writer.writerow([gold_triple, instance_triples, mask_position, type_triple])
