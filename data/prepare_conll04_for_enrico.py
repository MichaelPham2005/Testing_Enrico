import json
from datasets import load_dataset
from tqdm import tqdm

def convert_conll04_to_enrico_format(dataset_split):
    
    enrico_formatted_data = []
    
    for entry in tqdm(dataset_split, desc=f"Processing {dataset_split.split} split"):
        
        # --- 1. Chuyển đổi 'entities' thành 'spans' ---
        enrico_spans = []
        for entity in entry['entities']:
            # Quan trọng: 'end' trong nguồn là exclusive, nên cần trừ đi 1
            start_idx = entity['start']
            end_idx = entity['end'] - 1
            enrico_spans.append([start_idx, end_idx])

        # --- 2. Chuyển đổi 'relations' ---
        enrico_relations = []
        for rel in entry['relations']:
            head_span_index = rel['head']
            tail_span_index = rel['tail']
            
            relation_type = rel['type'] 
            
            enrico_relations.append([head_span_index, tail_span_index, relation_type])
            
        # --- 3. Tạo mục dữ liệu hoàn chỉnh cho EnriCo ---
        enrico_formatted_data.append({
            "tokenized_text": entry['tokens'],
            "spans": enrico_spans,
            "relations": enrico_relations
        })

    return enrico_formatted_data

if __name__ == "__main__":
    print("Tải bộ dữ liệu CoNLL-04 từ Hugging Face...")
    # Sử dụng tên dataset chính xác từ Hugging Face Hub
    dataset = load_dataset("DFKI-SLT/conll04")
    
    # Xử lý và lưu từng phần của bộ dữ liệu
    for split_name in ['train', 'validation', 'test']:
        output_filename = f"conll04_{split_name}_enrico_format.json"
        
        print(f"\nBắt đầu chuyển đổi tập {split_name}...")
        
        processed_split = convert_conll04_to_enrico_format(dataset[split_name])
        
        print(f"Lưu dữ liệu đã định dạng cho EnriCo vào tệp '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_split, f, ensure_ascii=False, indent=2)

    print("\nChuyển đổi hoàn tất!")
    print("Đã tạo ra 3 tệp dữ liệu huấn luyện theo định dạng của EnriCo.")