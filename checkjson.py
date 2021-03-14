import json


file = json.load(open("out.pkl.json","r"))

out = {
    "poses":[],
    "trans":[]
}


for i in file["poses"]:
    x = []
    for j in i:
        y = []
        for k in j:
            y.append(k)
        x.append(y)
    out["poses"].append(x)
for i in file["trans"]:
    x = []
    for j in i:
        x.append(j)
    out["trans"].append(x)
print(out)
json.dump(out,open("out2.pkl.json","w"))