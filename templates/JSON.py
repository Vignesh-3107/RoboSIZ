# import json

# with open("OCRLog.json","r") as file:
#     file = json.load(file)
# datum = {'_Text': 'SKL0125GH', '_CER': '10', '_Time': '10:25:09'}
# new = {}
# new[str(len(file)+1)] = datum
# file.update(new)
# with open("OCRLog.json", "w") as refile:
#     json.dump(file, refile,indent=4)

# import fastwer
# # Define reference text and output text
# ref = 'my name is kenneth'
# output = 'myy nime iz kenneth'

# # Obtain Sentence-Level Character Error Rate (CER)
# out = fastwer.score_sent(output, ref, char_level=True)
# print(out)