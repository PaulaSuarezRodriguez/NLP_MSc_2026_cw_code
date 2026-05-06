from BioRED_python_download.biored import BioredDataset
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification, DataCollatorWithPadding
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
import itertools


# Select existing config
builder = BioredDataset(config_name="biored_bigbio_kb")

# Download and prepare the dataset
builder.download_and_prepare()

# Load dataset splits
dataset = builder.as_dataset()

train = dataset["train"]
test = dataset["test"]
val = dataset["validation"]

print("Column names:\n", train.column_names)
print("Features:\n", train.features)
print("Entities:\n", train.features["entities"])
print("Number of columns:", train.num_columns)

print("First data sample in train:\n", train[0])

sample = train[0]
print(f"Title/Abstract: {sample['passages'][0]['text']}")
print(f"First Entity: {sample['entities'][0]}")
print(f"First Relation: {sample['relations'][0]}")

'''print("First data sample in train (line by line):")
print(train[0]["id"])
print(train[0]["document_id"])
print(train[0]["passages"])
print(train[0]["entities"])
print(train[0]["events"])
print(train[0]["coreferences"])
print(train[0]["relations"])'''

print("Size of training subset (i.e. number of data samples):", train.num_rows)
print("Size of validation subset (i.e. number of data samples):", val.num_rows)
print("Size of testing subset (i.e. number of data samples):", test.num_rows)






# 1. Initialize Tokenizer (BioBERT is standard for BioRED)
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', add_prefix_space=True)

# Map BioRED entity types to ID integers
label_list = [ "O",
               "B-ChemicalEntity", "I-ChemicalEntity",
                "B-DiseaseOrPhenotypicFeature","I-DiseaseOrPhenotypicFeature",
                "B-GeneOrGeneProduct","I-GeneOrGeneProduct",
                "B-SequenceVariant","I-SequenceVariant",
                "B-OrganismTaxon","I-OrganismTaxon",
                "B-CellLine", "I-CellLine"
            ]#["O", "B-Gene", "I-Gene", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]  # Simplified for example. #["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "ChemicalEntity", "SequenceVariant", "OrganismTaxon", "CellLine"]
label_to_id = {l: i for i, l in enumerate(label_list)}

'''model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-uncased",
    num_labels=len(label_list),
    id2label={i: l for i, l in enumerate(label_list)},
    label2id=label_to_id
)'''

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def preprocess_biored_for_ner(examples):
    # BioRED KB has 'passages' containing the full text
    #texts = [" ".join(p["text"]) for p in examples["passages"]]
    texts = [
        " ".join(
            passage_text
            for passage in p
            for passage_text in passage["text"]
        )
        for p in examples["passages"]
    ]

    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True  # CRITICAL for alignment
    )

    labels = []

    for i, document_entities in enumerate(examples["entities"]):
        offset_mapping = tokenized_inputs["offset_mapping"][i]
        # Initialize all labels as 'O' (0)
        doc_labels = np.zeros(len(offset_mapping), dtype=int)

        for entity in document_entities:
            start_char = entity["offsets"][0][0]
            end_char = entity["offsets"][0][1]
            entity_type = entity["type"]
            if entity_type not in [
                "ChemicalEntity",
                "DiseaseOrPhenotypicFeature",
                "GeneOrGeneProduct",
                "SequenceVariant",
                "OrganismTaxon",
                "CellLine"
            ]:
                print(entity_type + " " + str(start_char) + " " + str(end_char))

            # Map tokens to labels based on character offsets
            #first_token_found = False
            inside = False
            for idx, (start, end) in enumerate(offset_mapping):
                if start is None or start == end: #if start == end:  # Skip special tokens like [CLS] or [PAD]
                    doc_labels[idx] = -100  # Standard ignore index for loss
                    continue

                #if start >= start_char and end <= end_char:
                if not (end <= start_char or start >= end_char):
                    # Logic for B- and I- tags
                    #prefix = "B-" if not first_token_found else "I-"
                    '''if start == start_char:
                        prefix = "B-" if not first_token_found else "I-"
                    else:
                        prefix = "I-"'''
                    if not inside:
                        prefix = "B-"
                        inside = True
                    else:
                        prefix = "I-"
                    # Simplification: mapping all entities to Gene/Disease/Chem categories
                    '''clean_type = "Gene" if "Gene" in entity_type else "Disease" if "Disease" in entity_type else "Chemical"
                    label_str = f"{prefix}{clean_type}"'''
                    label_str = f"{prefix}{entity_type}"

                    '''doc_labels[idx] = label_to_id.get(label_str, 0)'''
                    if label_str in label_to_id:
                        doc_labels[idx] = label_to_id[label_str]
                    else:
                        raise ValueError(f"Unknown label: {label_str}")

                    #first_token_found = True

        labels.append(doc_labels.tolist())

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def preprocess_biored_for_re(example):
    entities = example["entities"]
    existing_relations = {(r["arg1_id"], r["arg2_id"]): r["type"] for r in example["relations"]}

    re_samples = []
    #TODO: Commented out alternative to be tried out if for training current implementation of method returning a dictionary in the way it currently does (even though as per HuggingFace it does need to return a dictionary) causes problems
    '''texts = []
    ent1_list = []
    ent2_list = []
    labels = []

    context = " ".join(example["passages"][0]["text"])'''
    # Generate every possible combination of 2 entities in the document
    for ent1, ent2 in itertools.combinations(entities, 2):
        pair = (ent1["id"], ent2["id"])
        reverse_pair = (ent2["id"], ent1["id"])

        # Determine label (Positive relation or 'None' for Negative Sampling)
        relation_type = "None"
        if pair in existing_relations:
            relation_type = existing_relations[pair]
        elif reverse_pair in existing_relations:
            relation_type = existing_relations[reverse_pair]

        # Create a text representation with Entity Markers
        # "Gene [E1] Factor-6 [/E1] is linked to [E2] Diabetes [/E2]"
        context = " ".join(example["passages"][0]["text"])  # Simple abstract-level context

        re_samples.append({
            "text": context,
            "entity_1": ent1["text"][0],
            "entity_2": ent2["text"][0],
            "label": relation_type
        })
    return {"re_data": re_samples}
    #TODO: Commented out alternative to be tried out if for training current implementation of method returning a dictionary in the way it currently does (even though as per HuggingFace it does need to return a dictionary) causes problems
    '''texts.append(context)
        ent1_list.append(ent1["text"][0])
        ent2_list.append(ent2["text"][0])
        labels.append(relation_type)

    return {
        "text": texts,
        "entity_1": ent1_list,
        "entity_2": ent2_list,
        "label": labels
    }'''

# Usage
# re_dataset = train.map(preprocess_biored_for_re)




if __name__ == "__main__":
    train_ner = train.map(preprocess_biored_for_ner, batched=True) # apply to train subset
    val_ner = val.map(preprocess_biored_for_ner, batched=True)  # apply to val subset
    test_ner = test.map(preprocess_biored_for_ner, batched=True)  # apply to test subset

    #print(train_ner[0])
    #print("NER data preprocessing complete")

    train_re = train.map(preprocess_biored_for_re) #, batched=True) # apply to train subset
    val_re = val.map(preprocess_biored_for_re) #, batched=True)  # apply to val subset
    test_re = test.map(preprocess_biored_for_re) #, batched=True)  # apply to test subset

    #print(train_re[0])
    #print("RE data preprocessing complete")

    train_ner.set_format(type="torch") # convert it to tensor for torch and Transformers to be able to handle the training subset for NER
    val_ner.set_format(type="torch") # convert it to tensor for torch and Transformers to be able to handle the val subset for NER
    test_ner.set_format(type="torch") # convert it to tensor for torch and Transformers to be able to handle the test subset for NER

    train_re.set_format(type="torch") # convert it to tensor for torch and Transformers to be able to handle the training subset for RE
    val_re.set_format(type="torch") # convert it to tensor for torch and Transformers to be able to handle the val subset for RE
    test_re.set_format(type="torch") # convert it to tensor for torch and Transformers to be able to handle the test subset for RE










