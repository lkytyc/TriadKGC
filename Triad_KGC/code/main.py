import asyncio
import csv
import json

from Triad_KGC.code.kg_completion.ontology_evolution import OntologyEvolutionController
from Triad_KGC.code.kg_verification.main import read_instance_triples_from_csv, KGVerification
from Triad_KGC.code.schema_exploration.schema_exploration import SchemaConstruction
from Triad_KGC.code.types.kg_operator import KGOperator
from Triad_KGC.code.utils.schema_parse.schema_process import schema_process


class TriadKGC:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.output_completed_triples_csv_path = f"../output/{dataset}_completed_triples.csv"
        self.output_verified_triples_csv_path = f"../output/{dataset}_verified_triples.csv"

    async def run_schema_exploration(self):
        # Initialize schema extractor
        extractor = SchemaConstruction(dataset=self.dataset)

        # Load input triples
        with open(f"../../data/{self.dataset}/test.json", "r", encoding='utf-8') as f:
            triples = json.load(f)

        with open(f"../../data/{self.dataset}/entities.json", 'r', encoding='utf-8') as f:
            entity_description = json.load(f)

        kg, schema_definition = await extractor.extract_from_kg(triples=triples)

        kg_json, user_add_schema_json = await schema_process(kg_data=json.loads(kg),
                                                             type_definition=schema_definition,
                                                             entity_description=entity_description)
        return kg_json, user_add_schema_json

    async def run_kg_completion(self, kg_json, user_add_schema_json):
        with open(f"../../data/{self.dataset}/dataset.json", "r", encoding="utf-8") as f:
            dateset_kg = json.load(f)
        with open(f"../../data/{self.dataset}/entities.json", "r", encoding="utf-8") as f:
            dateset_entities_desc = json.load(f)

        kg_operator = await KGOperator.from_json(json_str=kg_json)
        schema_kg_operator = await KGOperator.from_json(json_str=user_add_schema_json)
        ontology_evolution_controller = OntologyEvolutionController(kg_operator, schema_kg_operator)

        all_complete_triples = await ontology_evolution_controller.execute_evolution(dateset_kg, dateset_entities_desc)

        with open(self.output_completed_triples_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Gold Triple", "Completed Triples", "Mask Position", "Type Triple"])
            for gold_triple, completed_triples_str, mask_position in all_complete_triples:
                gold_str = f"({gold_triple.source_entity.name}, {gold_triple.edge.name}, {gold_triple.target_entity.name})"
                type_str = f"({gold_triple.source_entity.type}, {gold_triple.edge.type}, {gold_triple.target_entity.type})"
                if gold_str and completed_triples_str:
                    writer.writerow([gold_str, completed_triples_str, mask_position, type_str])

    async def run_kg_verification(self):
        all_triples = read_instance_triples_from_csv(file_path=self.output_completed_triples_csv_path)
        kg_verification = KGVerification()
        # Verify KG
        verified_triples = await kg_verification.verify_kg(unvalidated_triples=all_triples)

        # Step 1: Build mapping from completed triples to gold triples
        completed_to_gold_mapping = {}
        gold_to_position_mapping = {}
        with open(self.output_completed_triples_csv_path, 'r', encoding='utf-8') as f:
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
            instance_triples = instance_triples[:-2]
            csv_result[instance_triples] = type_triple

        # Step 3: Write final CSV with gold triples
        with open(self.output_verified_triples_csv_path, 'w', encoding='utf-8', newline='') as f:
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

    async def run(self):
        kg_json, user_add_schema_json = await self.run_schema_exploration()
        await self.run_kg_completion(kg_json=kg_json, user_add_schema_json=user_add_schema_json)
        await self.run_kg_verification()


if __name__ == "__main__":
    dataset = "WN18RR"

    triad_kgc = TriadKGC(dataset=dataset)

    asyncio.run(triad_kgc.run())
