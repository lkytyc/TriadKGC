import json

from Triad_KGC.code.config import settings
from Triad_KGC.code.types.kg_operator import KGOperator
from tqdm import tqdm
from Triad_KGC.code.utils.schema_parse.schema_exploration import extract_definition, get_new_entity_types_from_response, get_new_relationship_types_from_response, \
    convert_to_type_triples, extract_unique_entities_and_relations, \
    merge_type_dicts_with_semantic, delete_irrelevant_definitions, \
    convert_to_type_triples_without_type_triples, add_relationship_definition
from Triad_KGC.code.agents.agent_factory import AgentFactory

# Initialize the agents
entity_type_define_agent = AgentFactory.create_agent("schema_exploration", "EntityTypeDefineAgent")
relation_type_define_agent = AgentFactory.create_agent("schema_exploration", "RelationTypeDefineAgent")
entity_classify_agent = AgentFactory.create_agent("schema_exploration", "EntityClassifyAgent")
relation_classify_agent = AgentFactory.create_agent("schema_exploration", "RelationClassifyAgent")


class SchemaConstruction:
    def __init__(self, dataset):
        """
        Initialize SchemaConstruction with initial knowledge graph schema and definitions

        Args:
            type_triples: Triple with entity type and relationship type attached
            definition: Dictionary to store type definitions
        """
        self.dataset = dataset
        self.definition = {}
        self.type_triples = []
        self.entity_type_dict = {}  # Dictionary to store entity types and their instances
        self.relation_type_dict = {}  # Dictionary to store relation types and their instances

    async def extract_from_kg(
            self,
            triples,
            batch_size: int = settings.SCHEMA_EXPLORE_BATCH_SIZE
    ):
        """
        Process knowledge graph triples in batches with progress tracking

        Args:
            triples: List of triples to process
            api_key: API key for LLM service
            base_url: Base URL for API endpoint
            model: Model name to use
            batch_size: Number of triples to process per batch (default: 10)
        """
        total_triples = len(triples)
        total_batches = (total_triples + batch_size - 1) // batch_size

        # Initialize progress bar with description
        with tqdm(
                total=total_batches,
                desc="[Building Type KG]",  # Task description
                unit="batch",
                bar_format="{l_bar}{bar:20}{r_bar}",  # Control progress bar length
                colour="green"  # Progress bar color (optional)
        ) as pbar:
            for i in range(0, total_triples, batch_size):
                batch = triples[i:i + batch_size]
                current_batch = i // batch_size + 1

                # Update progress bar information
                pbar.set_postfix({
                    "Current Batch": f"{current_batch}/{total_batches}",
                    "Triple Count": len(batch)
                })

                # Process current batch
                await self.extract_type_triples(
                    triples=batch
                )
                pbar.update(1)  # Update progress

        # Post-processing
        tqdm.write(
            f"\n✅ Processing complete! Processed {total_triples} triples, resulting in {len(self.type_triples)} type triples")
        kg_with_type = await self.contruct_schema_graph(self.type_triples)
        await self.type_definition()
        self.definition = delete_irrelevant_definitions(kg_with_type, self.definition)
        if self.dataset == "WN18RR":
            self.definition = add_relationship_definition(self.definition)
        return kg_with_type, self.definition

    async def contruct_schema_graph(self, type_triples):
        """
        Construct schema graph from type triples

        Args:
            type_triples: List of type triples to construct graph from

        Returns:
            JSON representation of the constructed knowledge graph
        """
        op = KGOperator()
        for type_triple in type_triples:
            # Create nodes if they don't exist
            if not await op.find_node_by_name(type_triple["DirectionalEntity"]["name"]):
                node = await op.create_node(type_=type_triple["DirectionalEntity"]["type"],
                                            name=type_triple["DirectionalEntity"]["name"])
            if not await op.find_node_by_name(type_triple["DirectedEntity"]["name"]):
                await op.create_node(type_=type_triple["DirectedEntity"]["type"],
                                     name=type_triple["DirectedEntity"]["name"])

            # Find nodes and create relationship
            node1 = await op.find_node_by_name(type_triple["DirectionalEntity"]["name"])
            node2 = await op.find_node_by_name(type_triple["DirectedEntity"]["name"])
            await op.create_edge(str(node1.id), str(node2.id), type_triple["Relation"]["type"],
                                 type_triple["Relation"]["name"])

        # Convert graph to JSON format
        kg_with_type = await op.to_json()
        return kg_with_type

    async def type_definition(self):
        """
        Generate definitions for entity and relation types with dual progress bars

        Args:
            api_key: API key for LLM service
            base_url: Base URL for API endpoint
            model: Model name to use
        """
        # Entity type definition task
        entity_items = list(self.entity_type_dict.items())
        with tqdm(
                total=len(entity_items),
                desc="[Entity Type Definition]",  # Task description
                unit="type",
                colour="green",  # Progress bar color
                bar_format="{l_bar}{bar:20}{r_bar}"
        ) as pbar_entity:
            for key, values in entity_items:
                values_str = ", ".join(values)
                entity_type_dict_string = f"{key}: {values_str}"

                # Get API response
                entity_definition_response = await entity_type_define_agent.run(entity_type_dict_string=entity_type_dict_string)

                # Extract definition
                self.definition.update(extract_definition(entity_definition_response))
                pbar_entity.update(1)  # Update progress
                pbar_entity.set_postfix({"Current Entity Type": key})  # Show current type being processed

        if self.dataset == "fb15k237":
            # Relation type definition task
            relation_items = list(self.relation_type_dict.items())
            with tqdm(
                    total=len(relation_items),
                    desc="[Relation Type Definition]",  # Task description
                    unit="type",
                    colour="blue",  # Progress bar color
                    bar_format="{l_bar}{bar:20}{r_bar}"
            ) as pbar_relation:
                for key, values in relation_items:
                    values_str = ", ".join(values)
                    relation_type_dict_string = f"{key}: {values_str}"

                    # Get API response
                    relation_definition_response = await relation_type_define_agent.run(
                        relation_type_dict_string=relation_type_dict_string)

                    # Extract definition
                    self.definition.update(extract_definition(relation_definition_response))
                    pbar_relation.update(1)  # Update progress
                    pbar_relation.set_postfix({"Current Relation Type": key})  # Show current relation being processed

        tqdm.write("✅ Type definition generation complete!")

    async def extract_type_triples(self, triples):
        """
        Extract schema information from a batch of triples

        Args:
            triples: List of triples to process
            api_key: API key for LLM service
            base_url: Base URL for API endpoint
            model: Model name to use
            batch_size: Expected number of triples in this batch
        """
        # Extract unique entities and relations
        entities_and_relations = extract_unique_entities_and_relations(triples)
        entity_string = ", ".join(entities_and_relations["entities"])
        relation_string = ", ".join(entities_and_relations["relations"])

        # Classify entities
        entity_classify_response = await entity_classify_agent.run(entity_string=entity_string)

        # Update entity type dictionary
        if not self.type_triples:
            self.entity_type_dict = get_new_entity_types_from_response(response=entity_classify_response)
        else:
            temp_entity_type_dict = get_new_entity_types_from_response(response=entity_classify_response)
            self.entity_type_dict = await merge_type_dicts_with_semantic(self.entity_type_dict,
                                                                         temp_entity_type_dict)

        type_triples = []
        if self.dataset == "fb15k237":
            # Classify relations
            relation_classify_response = await relation_classify_agent.run(relation_string=relation_string)

            # Update relation type dictionary
            if not self.type_triples:
                self.relation_type_dict = get_new_relationship_types_from_response(response=relation_classify_response)
            else:
                temp_relation_type_dict = get_new_relationship_types_from_response(response=relation_classify_response)
                self.relation_type_dict = await merge_type_dicts_with_semantic(self.relation_type_dict,
                                                                               temp_relation_type_dict)

            # Convert to type triples
            type_triples = convert_to_type_triples(triples, self.entity_type_dict, self.relation_type_dict)
        elif self.dataset == "WN18RR":
            type_triples = convert_to_type_triples_without_type_triples(triples, self.entity_type_dict)

        # Check for data loss in this batch
        if len(type_triples) == len(triples):
            self.type_triples.extend(type_triples)
        else:
            # If data was lost, reprocess this batch
            await self.extract_type_triples(triples)
