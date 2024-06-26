import json

def print_evaluation_results(results_file):
    with open(results_file, 'r') as file:
        results = json.load(file)

    print("Intent Evaluation Report")
    print("------------------------")
    for intent, metrics in results['intent_evaluation']['report'].items():
        if intent != 'micro avg' and intent != 'macro avg':
            print(f"Intent: {intent}")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1-Score: {metrics['f1-score']:.2f}")
            print()

# Print evaluation results
print_evaluation_results('results/intent_report.json')
