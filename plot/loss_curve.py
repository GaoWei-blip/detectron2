import json
import matplotlib.pyplot as plt


parsed=[]
with open('output/metrics.json') as f:
    try:
        for line in f:
            parsed.append(json.loads(line))
    except:
        print("json format is not corrrect")
        exit(1)

# Extract relevant data
parsed = parsed[:-1]
iterations = []
loss_values = []

for entry in parsed:
    try:
        iterations.append(entry['iteration'])
        loss_values.append(entry['total_loss'])
    except:
        print("total_loss not in entry")



# Plot the loss curve
plt.plot(loss_values, label='Total Loss')
plt.title('Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Total Loss')
plt.legend()
plt.show()
