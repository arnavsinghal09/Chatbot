import json
import yaml

def convert_json_to_rasa(json_file, nlu_file, domain_file, stories_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    nlu_data = {"nlu": []}
    domain_data = {
        "intents": [],
        "responses": {},
        "session_config": {
            "session_expiration_time": 60,
            "carry_over_slots_to_new_session": True
        }
    }
    stories_data = {"stories": []}

    for item in data['intents']:
        intent = item['tag']
        examples = "\n- ".join(item['patterns'])
        
        nlu_data['nlu'].append({
            "intent": intent,
            "examples": f"- {examples}"
        })
        
        domain_data['intents'].append(intent)
        domain_data['responses'][f"utter_{intent}"] = [{"text": response} for response in item['responses']]
        
        stories_data['stories'].append({
            "story": f"{intent} path",
            "steps": [
                {"intent": intent},
                {"action": f"utter_{intent}"}
            ]
        })
    
    with open(nlu_file, 'w') as file:
        yaml.dump(nlu_data, file, default_flow_style=False)
    
    with open(domain_file, 'w') as file:
        yaml.dump(domain_data, file, default_flow_style=False)
    
    with open(stories_file, 'w') as file:
        yaml.dump(stories_data, file, default_flow_style=False)

# Convert intents.json to Rasa files
convert_json_to_rasa('intents.json', 'data/nlu.yml', 'domain.yml', 'data/stories.yml')
