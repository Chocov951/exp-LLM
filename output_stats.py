import re
import json
import ast

def parse_output_file(file_path):
    """Parse the output file and extract statistics about generated responses."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract bm25_topk from the first line
    first_line = content.split('\n')[0]
    bm25_topk_match = re.search(r'bm25_topk=(\d+)', first_line)
    if not bm25_topk_match:
        raise ValueError("Could not find bm25_topk in the first line")
    
    # Extract dataset name from the first line
    dataset_match = re.search(r'dataset=\'(\w+)\'', first_line)
    if not dataset_match:
        raise ValueError("Could not find dataset in the first line")
    dataset_name = dataset_match.group(1)
    print(f"Dataset: {dataset_name}")

    bm25_topk = int(bm25_topk_match.group(1))
    print(f"BM25 K: {bm25_topk}")
    
    # Find all generated responses
    response_pattern = r'Generated response : (\[.*?\n)'
    responses = re.findall(response_pattern, content, re.DOTALL)

    nb_responses = len(responses)
    print(f"\nNumber of generated responses: {nb_responses}")

    # Analyze each response
    responses_ending_with_bracket = 0
    response_lengths = []
    invalid_elements_count = []
    repeated_elements_count = []
    
    for i, response_str in enumerate(responses):
        # Check if response ends with closing bracket
        if response_str.strip().endswith(']'):
            responses_ending_with_bracket += 1
        else:
            response_str = response_str.rsplit(',', 1)[0] + ']'
        
        try:
            # Parse the response as a list
            response_list = ast.literal_eval(response_str)
            
            # Convert string elements to integers and calculate statistics
            int_list = []
            invalid_count = 0
            
            for item in response_list:
                try:
                    num = int(item)
                    int_list.append(num)
                    # Check if number is outside valid range [0, bm25_topk-1]
                    if num < 0 or num >= bm25_topk:
                        invalid_count += 1
                except (ValueError, TypeError):
                    invalid_count += 1
            
            # Count repeated elements
            unique_elements = set(int_list)
            repeated_count = len(int_list) - len(unique_elements)
            
            response_lengths.append(len(int_list))
            invalid_elements_count.append(invalid_count)
            repeated_elements_count.append(repeated_count)
            
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing response {i+1}: {e}")
            response_lengths.append(0)
            invalid_elements_count.append(0)
            repeated_elements_count.append(0)
    
    # Calculate statistics
    avg_length = sum(response_lengths) / nb_responses if nb_responses > 0 else 0
    avg_invalid_elements = sum(invalid_elements_count) / nb_responses if nb_responses > 0 else 0
    avg_repeated_elements = sum(repeated_elements_count) / nb_responses if nb_responses > 0 else 0

    print(f"Responses ending with ']': {responses_ending_with_bracket}")
    print(f"Average length of generated lists: {avg_length:.2f}")
    print(f"Average number of invalid elements per list: {avg_invalid_elements:.2f}")
    print(f"Average number of repeated elements per list: {avg_repeated_elements:.2f}")
    # print(f"(Invalid elements are < 0 or >= {bm25_topk})")
    
    # Additional statistics
    if response_lengths:
        print(f"\nAdditional statistics:")
        print(f"Min list length: {min(response_lengths)}")
        print(f"Max list length: {max(response_lengths)}")
        print(f"Total invalid elements across all responses: {sum(invalid_elements_count)}")
        print(f"Total repeated elements across all responses: {sum(repeated_elements_count)}")
        # print(f"Min repeated elements in a list: {min(repeated_elements_count)}")
        print(f"Max repeated elements in a list: {max(repeated_elements_count)}")
    
    return {
        'bm25_topk': bm25_topk,
        'num_responses': len(responses),
        'responses_ending_with_bracket': responses_ending_with_bracket,
        'avg_length': avg_length,
        'avg_invalid_elements': avg_invalid_elements,
        'avg_repeated_elements': avg_repeated_elements,
        'response_lengths': response_lengths,
        'invalid_elements_count': invalid_elements_count,
        'repeated_elements_count': repeated_elements_count
    }

if __name__ == "__main__":
    import sys
    
    # Parse the output example file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "output_exemple.txt"
    
    try:
        stats = parse_output_file(file_path)
        # print("\n" + "="*50)
        # print("SUMMARY:")
        # print(f"BM25 Top-K: {stats['bm25_topk']}")
        # print(f"Total responses: {stats['num_responses']}")
        # print(f"Responses ending with ']': {stats['responses_ending_with_bracket']}")
        # print(f"Average list length: {stats['avg_length']:.2f}")
        # print(f"Average invalid elements: {stats['avg_invalid_elements']:.2f}")
        # print(f"Average repeated elements: {stats['avg_repeated_elements']:.2f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        print("Make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")