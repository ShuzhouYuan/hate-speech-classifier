import json

tweets = []
for line in open('./amateur_expert.json', 'r'):
    tweets.append(json.loads(line))

print(type(tweets[0]))