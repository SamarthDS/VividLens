crime_prompt_map = {
    "vandalism": [
        "A person is spray painting a wall",
        "A person is breaking public property",
        "Someone is damaging a car or window"
    ],
    "stealing": [
        "A person is stealing something from a bag",
        "Someone is pickpocketing",
        "Someone is taking someone else's belongings"
    ],
    "shoplifting": [
        "A person is hiding an item inside their clothes in a store",
        "Someone is leaving a store without paying",
        "Shoplifting is taking place"
    ],
    "shooting": [
        "A person is firing a gun",
        "A shooting incident is occurring",
        "Someone is aiming and shooting at another person"
    ],
    "robbery": [
        "A person is robbing a store",
        "Someone is threatening another person to steal",
        "A robbery is happening"
    ],
    "roadaccidents": [
        "Two vehicles are colliding",
        "A person is running",
        "There is a traffic accident"
    ],
    "fighting": [
        "Two people are fighting each other",
        "A physical altercation is occurring",
        "People are engaged in a fistfight"
    ],
    "explosion": [
        "An explosion is taking place",
        "A bomb has exploded",
        "A blast is occurring"
    ],
    "burglary": [
        "Someone is breaking into a house",
        "A burglary is happening at night",
        "A thief is entering through a window"
    ],
    "assault": [
        "A person is being physically assaulted",
        "An attacker is punching someone",
        "Someone is hitting another person"
    ],
    "arson": [
        "Someone is setting fire to a building",
        "Arson is happening",
        "A person is starting a fire intentionally"
    ],
    "arrest": [
        "Police are arresting a suspect",
        "Someone is being handcuffed",
        "A police officer is detaining someone"
    ],
    "abuse": [
        "Someone is being abused",
        "A person is yelling and hitting another person",
        "Domestic violence is occurring"
    ],
    "normal": [
        "Everything looks normal",
        "There is no unusual activity",
        "People are walking peacefully"
    ]
}

# Flatten all prompts into a single list for ActionCLIP
crime_prompts = [prompt for prompts in crime_prompt_map.values() for prompt in prompts]

# Individual prompt lists for each crime type
robbery_prompts = crime_prompt_map["robbery"]
fighting_prompts = crime_prompt_map["fighting"]
shooting_prompts = crime_prompt_map["shooting"]
vandalism_prompts = crime_prompt_map["vandalism"]
stealing_prompts = crime_prompt_map["stealing"]
shoplifting_prompts = crime_prompt_map["shoplifting"]
arson_prompts = crime_prompt_map["arson"]
abuse_prompts = crime_prompt_map["abuse"]
arrest_prompts = crime_prompt_map["arrest"]
explosion_prompts = crime_prompt_map["explosion"]
burglary_prompts = crime_prompt_map["burglary"]
road_accident_prompts = crime_prompt_map["roadaccidents"]

# Helper function to get prompts by incident name

def get_prompts_for_filename(filename):
    name = filename.lower()
    if "robbery" in name:
        return robbery_prompts
    elif "fighting" in name:
        return fighting_prompts
    elif "shooting" in name:
        return shooting_prompts
    elif "vandalism" in name:
        return vandalism_prompts
    elif "stealing" in name:
        return stealing_prompts
    elif "shoplifting" in name:
        return shoplifting_prompts
    elif "arson" in name:
        return arson_prompts
    elif "abuse" in name:
        return abuse_prompts
    elif "arrest" in name:
        return arrest_prompts
    elif "explosion" in name:
        return explosion_prompts
    elif "burglary" in name:
        return burglary_prompts
    elif "accident" in name or "crash" in name:
        return road_accident_prompts
    else:
        return crime_prompts
