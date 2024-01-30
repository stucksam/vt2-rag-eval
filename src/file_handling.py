import json

from src.question_generator import fix_transcripts_ner


def combine_ragas_generated_datasets():
    def fix_dicts(content: list):
        con = []
        for entry in content:
            if isinstance(entry["context"], list):
                entry["context"] = fix_transcripts_ner(' '.join(entry["context"]))
            entry["context"] = entry["context"].replace("- ", "").replace(".\n", ". ").strip()
            if isinstance(entry["answer"], list):
                entry["answer"] = fix_transcripts_ner(" ".join(entry["answer"]).strip())
            if isinstance(entry["question"], list):
                entry["question"] = fix_transcripts_ner(" ".join(entry["question"]).strip())
            con.append(entry)
        return con

    data_list = [
        {
            'question': "Who are the characters we met in Episode 1 of Baldur's Gate 3 Let's Play Series?",
            'context': [
                "- In Episode 1, we were captured by a nautiloid ship and infected with a parasite in our brain, which we learned will ultimately turn us into a mind flayer if we don't do anything about it.",
                "- We also met several characters as we wandered the material plane after that nautiloid ship crashed, which includes Starion, Shadowheart, and Gale.",
                "- Lae’zel is currently back at our camp.",
                "- We are currently searching for a way to be cured of this parasite.",
                "- We did hear that there's a Healer named Nettie in a camp nearby, so I think we're going to pursue that.",
                "- But Lae’zel has also said that she could bring us to a Gethianki crash where her people would purify us, whatever that means.",
                "- We did discover some ruins in Episode 1, but didn't fully explore them.",
                "- So let's go ahead and head back to the ruins right here and actually go inside this time and see what's going on."],
            'answer': [
                "The characters we met in Episode 1 of Baldur's Gate 3 Let's Play Series are Starion, Shadowheart, and Gale."],
            'question_type': 'simple'
        },
        {
            'question': "What additional effect does the wearer's weapon have when they are healed?",
            'context': [
                "Whenever the wearer is healed, their weapon becomes coded in Magic and Deals an additional 1d6 poison, that's pretty nice.",
                "And I can pop a healing word or even take a potion, then you get that one to six poison damage."],
            'answer': [
                "The additional effect that the wearer's weapon has when they are healed is that it becomes coded in Magic and deals an additional 1d6 poison damage."],
            'question_type': 'simple'
        },
        {
            'question':
                "What is the significance of the reader's assistance in locating magic items for the narrator?",
            'context': ["- That is why I turn to you.",
                        "- I need you to help me find magic items to consume.",
                        "- It is vital, dare I say it, critical.",
                        "- Hmm, I fail to see why you need me to help you with this.",
                        "- A fair point, however, until recently, I was able to rely on a supply of artifacts stored in my tower in Waterdeep.",
                        "- A supply that has now run dry.",
                        "- The reality of the matter is that a lone wizard with a chronic impairment such as my own is not in the most ideal of situations with regards to self-defense.",
                        "- The manner of artifacts I need are not often found waiting patiently on a shopkeep's shelf.",
                        "- One usually has to lift them delicately from trap-filled tombs or prizing from the hands of violent ne'er-do-wells.",
                        "- The danger involved, or great cost.",
                        "- And why exactly would I risk either of those things for a wizard I barely know?",
                        "- Having a wizard like me around is quite the boon when facing the perils that stalk these lands.",
                        "- It'll be far harder for me to assist you if I can barely stand upright.",
                        "- To me, your help could be the difference between life and death."],
            'answer': [
                "The significance of the reader's assistance in locating magic items for the narrator is that the narrator, who is a wizard with a chronic impairment, needs these items for self-defense. The narrator used to have a supply of artifacts in their tower, but it has run dry. The items they need are not easily found in shops and usually require dangerous or costly efforts to obtain. The narrator believes that having these items is crucial for their survival, and the reader's help could be the difference between life and death for them."],
            'question_type': 'reasoning'
        },
        {
            'question': "What happens if the speaker doesn't consume the artifacts for their condition?",
            'context': [
                "- What it comes down to is this: every so often, I need to get my hands on a powerful magical item and absorb the weave inside.",
                "- And what happens if you don't consume these artifacts?",
                "- I'll spare you the finer details, but it begins with a simple biological deterioration, muscle spasms, disorientation, a slight ringing in the ears.",
                "- And if it's there for too long, it's deadly.",
                "- It's been days since I last consumed an artifact before we were abducted.",
                "- It's only a matter of time before my craving returns."],
            'answer': [
                "If the speaker doesn't consume the artifacts for their condition, they will experience biological deterioration, muscle spasms, disorientation, and a slight ringing in the ears. If this condition persists for too long, it can become deadly. The speaker mentions that it has been days since they last consumed an artifact, indicating that their craving for it will eventually return."],
            'question_type': 'reasoning'
        },
        {
            'question': ['What condition does the speaker have and what is the recommended treatment for it?',
                         'What are the potential consequences if the speaker does not consume the artifacts related to their condition?'],
            'context': [
                '- You see, I have this condition, very different from the parasite we share, but just as deadly.',
                '- Thank you for the offer, but the treatment for my condition is very specific.',
                '- What it comes down to is this: every so often, I need to get my hands on a powerful magical item and absorb the weave inside.',
                "What it comes down to is this: every so often, I need to get my hands on a powerful magical item and absorb the weave inside. \nAnd what happens if you don't consume these artifacts?"],
            'answer': [
                'The speaker has a specific condition that requires them to absorb the weave inside a powerful magical item as a treatment.',
                'The potential consequences if the speaker does not consume the artifacts related to their condition are not mentioned in the given context.'],
            'question_type': 'reasoning'
        },
        {
            'question': "What benefits does the spell Warding Bond offer and for how long?",
            'context': ['- "It gives pain resistance to damage and a plus one to their Armor class and saving throws."',
                        '- "And it lasts until a long rest, and it doesn\'t require concentration."'],
            'answer': [
                'The spell Warding Bond offers pain resistance to damage, a plus one to Armor class and saving throws. It lasts until a long rest and does not require concentration.'],
            'question_type': 'reasoning'
        },
        {
            'question': "Who are the characters in Episode 1 of the Baldur's Gate 3 Let's Play Series and how can we cure the brain parasite?",
            'context': [
                "In Episode 1, we were captured by a nautiloid ship and infected with a parasite in our brain, which we learned will ultimately turn us into a mind flayer if we don't do anything about it.",
                "We also met several characters as we wandered the material plane after that nautiloid ship crashed, which includes Starion, Shadowheart, Gale, and Lae’zel.",
                "Lae’zel is currently back at our camp.",
                "We are currently searching for a way to be cured of this parasite.",
                "We did hear that there's a Healer named Nettie in a camp nearby, so I think we're going to pursue that.",
                "But Lae’zel has also said that she could bring us to a Gethianki crash where her people would purify us, whatever that means.",
                "So we have a couple of different options here.",
                "But we did discover some ruins in Episode 1, but didn't fully explore them."],
            'answer': [
                "The characters in Episode 1 of the Baldur's Gate 3 Let's Play Series are Starion, Shadowheart, Gale, and Lae’zel. To cure the brain parasite, the options mentioned are seeking help from a healer named Nettie in a nearby camp or going with Lae’zel to a Gethianki crash where her people can purify them."],
            'question_type': "reasoning"
        },
        {
            'question': "In what ways do the Wood Elf subrace's racial traits, like darkvision and Fey ancestry, contribute to their archery and camouflage skills in Faerun's forests?",
            'context': ["- We get darkvision, which is a really good racial feature in this game.",
                        "- And also Fey ancestry, you have advantage on saving throws against being charmed and magic can't put you to sleep, which is also nice.",
                        "- And for our subrace, we're gonna go the Wood Elf.",
                        "- These elves spend their reclusive lives in Faerun's forests.",
                        "- Decades of training in archery and camouflage are enhanced by an otherworldly swiftness."],
            'answer': [
                "The Wood Elf subrace's racial traits, such as darkvision and Fey ancestry, contribute to their archery and camouflage skills in Faerun's forests in several ways. \n\nFirstly, darkvision allows Wood Elves to see in darkness as if it were dim light, giving them an advantage when navigating through the forest at night or in areas with low light. This ability helps them to spot potential targets or threats more easily, enhancing their archery skills.\n\nSecondly, Fey ancestry provides Wood Elves with advantage on saving throws against being charmed and immunity to being put to sleep by magic. This resistance to magical effects helps them to remain focused and alert while hunting or hiding, making them less susceptible to distractions or enchantments that could hinder their archery or camouflage abilities.\n\nAdditionally, the Wood Elf subrace's reclusive lives in Faerun's forests and their decades of training in archery and camouflage are further enhanced by an otherworldly swiftness. This swiftness, which is likely a result of their Fey ancestry, allows them to move quickly and silently through the forest, making it easier for them to find advantageous positions for archery or to blend seamlessly into their surroundings for camouflage.\n\nOverall, the combination of darkvision, Fey ancestry, and their otherworldly swiftness greatly contributes to the Wood Elf subrace's archery and camouflage skills in Faerun's forests, giving them a significant advantage in their reclusive lives and making them formidable hunters and scouts."],
            'question_type': 'conditional'
        },
        {
            'question': 'What is the consequence of accepting the offer of eternal life?',
            'context': [
                "- Given that my choices were eternal life or bleed to death on the street, I took him up on the offer.",
                "- It was only afterward I realized just how long eternity could be.",
                "- A spawn is less than a slave.",
                "- We have no choice but to obey our master's commands.",
                "- They speak and our bodies react.",
                "- It's all part of the deal."],
            'answer': [
                "The consequence of accepting the offer of eternal life is that the person becomes a slave and must obey their master's commands."],
            'question_type': 'simple'
        }
    ]
    data_list_2 = [
        {
            'question': [
                "What are the effects of casting Hunter's Mark as a hunter spell on the ranger's combat abilities and movement speed?"],
            'context': [
                '- "This is a really, really good hunter spell, especially earlier on in the game, and it does require concentration, but it only costs a bonus action and a spell slot."',
                '- "It\'s not an action, so I can cast Hunter\'s Mark on an enemy and then also attack that enemy in the same turn."',
                '- "And since I have low wisdom as a ranger, I don\'t really have good wisdom, I want to stick to spells that don\'t work off of my wisdom modifier."',
                '- "What I\'m going to take here is Longstrider, increase the creature\'s movement speed by 10 feet, which lasts until a long rest."',
                '- "And that kind of, you know it kind of messes with your ability to pass Hunter\'s Mark around as both require a bonus action, so I\'m going to take longstrider, accompany longstrider with my what elf plus five movement speed."',
                '- "My character is going to be going all over the battlefield and getting right into the enemy\'s faces."',
                '- "And then for fighting style, we have archery, defense, dueling, and two weapon fighting."',
                '- "And I\'m gonna go ahead and take defense, which gives us a plus one to our Armor class."'],
            'answer': [
                "Casting Hunter's Mark as a hunter spell has the following effects on the ranger's combat abilities and movement speed:\n- It allows the ranger to cast Hunter's Mark on an enemy and attack that enemy in the same turn, as it only costs a bonus action.",
                "- It requires concentration.",
                "- It does not rely on the ranger's wisdom modifier, making it a good choice for rangers with low wisdom.",
                "- It may limit the ranger's ability to pass Hunter's Mark around, as both casting the spell and using Longstrider require a bonus action.",
                "- Longstrider, when combined with the ranger's Wood Elf racial bonus, increases the creature's movement speed by 10 feet until a long rest.",
                "- The ranger's character will be highly mobile, moving all over the battlefield and getting close to the enemy.",
                "- The ranger has the option to choose from different fighting styles, such as archery, defense, dueling, and two weapon fighting. In this case, the ranger chooses defense, which provides a +1 bonus to their Armor class."],
            'question_type': 'multi_context'
        },
        {
            'question': ['What type of armor is the speaker planning to use for their Ranger character?'],
            'context': ['- "I am going to be using heavy two-handed weapons."',
                        '- "We did loot that burning blade from commander zulk on the ship so let\'s go ahead and put that on right now."',
                        '- "Let\'s go ahead and put on the scale now. The booster Armor class by quite a bit here."',
                        '- "I do plan on trying to find heavy armor as soon as I can."',
                        '- "Since we\'re focused on heavy armor that doesn\'t matter as much, but I do have to keep an eye out for heavy armor and try to find it as soon as I possibly can."'],
            'answer': ['The speaker is planning to use heavy armor for their Ranger character.'],
            'question_type': 'simple'
        },
        {
            'question': [
                'What advantages does the Grim Harvest feature offer to a paladin-rogue multiclass character?'],
            'context': ["- Our necromancy features Grim Harvest.",
                        "- Once per turn, if you kill a creature with a spell, you regain hit points equal to twice the spell slot level used.",
                        "- Thrice if it's a necromancy spell.",
                        "- Undead and constructs are unaffected.",
                        "- So your subclass choice really pushes you to want to use spells of that particular School of magic.",
                        "- Pretty cool feature Grim Harvest.",
                        "- We don't really have any necromancy Spells at the moment.",
                        "- We do have false life.",
                        "- That's not going to really help us with Grim Harvest though.",
                        "- And also let's just take a look at the multi-class scene because I haven't looked at that yet.",
                        "- This Advanced feature allows you to build powerful combinations of classes at the expense of higher level class features.",
                        "- You can only level up one class at a time.",
                        "- So we could just right away multi-class Asterion into another class, but I think I'm going to keep him make him a paladin half rogue."],
            'answer': [
                "The Grim Harvest feature offers the advantage of regaining hit points when a creature is killed with a spell. The amount of hit points regained is equal to twice the spell slot level used, or thrice if it's a necromancy spell. However, undead and constructs are unaffected by this feature. This feature encourages the use of spells from the necromancy School of magic. As for the paladin-rogue multiclass character, it is not mentioned how the Grim Harvest feature specifically benefits this combination."],
            'question_type': 'multi_context'
        }
    ]

    data_list = fix_dicts(data_list)
    data_list.extend(fix_dicts(data_list_2))
    with open("data/testset_ragas.json", "r", encoding="utf-8") as f:
        prev_ragas = json.loads(f.read())
    data_list.extend(fix_dicts(prev_ragas))
    with open("data/ragas_definitive.json", "w", encoding="utf-8") as f:
        json.dump(data_list, f)


def select_generated_questions():
    with open("data/qa_dataset_further.json", "r") as f:
        content = json.loads(f.read())
    final = []
    trash = []
    for entry in content:
        print(f"Q: {entry['query']}")
        print(f"A: {entry['answer']}")
        print(entry["context"] + "\n")
        decision = input("Keep: ")
        if decision == "y" or decision == "yes":
            final.append(entry)
        elif decision == "stop":
            with open("data/qa_dataset_further_final._oldjson", "a", encoding="utf-8") as f:
                json.dump(final, f)
            break
        else:
            trash.append(entry)

        with open("data/qa_dataset_further_final._oldjson", "a", encoding="utf-8") as f:
            json.dump(final, f)

        with open("data/qa_dataset_further_trash.json", "a", encoding="utf-8") as f:
            json.dump(trash, f)


def combine_different_datasets():
    with open("data/qa_dataset_usable.json", "r", encoding="utf-8") as f:
        content_usable = json.loads(f.read())

    with open("data/qa_dataset_further_final.json", "r", encoding="utf-8") as f:
        content_further = json.loads(f.read())

    with open("data/ragas_definitive.json", "r", encoding="utf-8") as f:
        content_ragas = json.loads(f.read())

    content = []
    for i, entry in enumerate(content_usable):
        ground_truth = fix_transcripts_ner(entry["ground_truth"]) if "ground_truth" in entry and entry["ground_truth"] not in ["n", "skip"] else None
        if "ground_truth" in entry and entry["ground_truth"] not in ["n", "skip"]:
            content.append({
                "question": fix_transcripts_ner(entry["query"]),
                "ground_truth": ground_truth,
                "context": fix_transcripts_ner(entry["context"]).strip(),
                "meta": {
                    "id_source_doc": entry["doc_id"],
                    "id_query": f"Q_G1_{i}",
                    "context_length": entry['context_length']
                }
            })
    for j, entry in enumerate(content_further):
        content.append({
            "question": fix_transcripts_ner(entry["query"]),
            "ground_truth": fix_transcripts_ner(entry["answer"]),
            "context": fix_transcripts_ner(entry["context"]).replace("- ", "").replace(".\n", ". ").strip(),
            "meta": {
                "id_source_doc": entry["doc_id"],
                "id_query": f"Q_G2_{j}",
                "context_length": entry['context_length']
            }
        })
    for k, entry in enumerate(content_ragas):
        content.append({
            "question": fix_transcripts_ner(entry["question"]),
            "ground_truth": fix_transcripts_ner(entry["ground_truth"]),
            "contexts": fix_transcripts_ner(entry["context"]).replace("- ", "").replace(".\n", ". ").strip(),
            "meta": {
                "id_source_doc": -1,
                "id_query": f"Q_R1_{k}",
                "ragas_q_type": entry["question_type"],
                "context_length": len(nltk.word_tokenize(entry["context"])),
            }
        })
    with open("data/qa_dataset_bg3_lp.json", "w", encoding="utf-8") as f:
        json.dump(content, f)
