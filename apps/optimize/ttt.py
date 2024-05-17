import re
import matplotlib.pyplot as plt

# Your input string
input_string = [
"Max 1 act: 24.766929408035875, Max 2 act: 36.46159086420851, Reward:-3.3599549615003435, Error vel: 0.03272942520532119",
"Max 1 act: 24.7803512302777, Max 2 act: 32.87674107170385, Reward:-2.9679065555992556, Error vel: 0.04042574011465826",
"Max 1 act: 32.32011634462596, Max 2 act: 36.68094260018092, Reward:-2.5583036054557646, Error vel: 0.04098222699073695",
"Max 1 act: 45.72761928020896, Max 2 act: 32.789537616949175, Reward:-2.147839949215124, Error vel: 0.05488055136487786",
"Max 1 act: 32.622422500681154, Max 2 act: 58.827528134137246, Reward:-1.7387147071838702, Error vel: 0.12225548115795093",
"Max 1 act: 39.86963538868039, Max 2 act: 31.292308995394595, Reward:-1.3200568895884037, Error vel: 0.20730669236699956",
"Max 1 act: 1766.4788928387588, Max 2 act: 1315.3510804895116, Reward:-0.9167614245692708, Error vel: 0.05892373618222441",
"Max 1 act: 205.68094623201296, Max 2 act: 224.49296224103762, Reward:-0.36340331620795296, Error vel: 1.2298073631372128",
"Max 1 act: 24.4102965153385, Max 2 act: 33.800637272696434, Reward:1.0133930393264787e-05, Error vel: 0.029589376368271417",

]
# Define the regex pattern for floating point numbers (including negative numbers)
pattern = r'-?\d+\.\d+'
res = {}
res["min_trq"] = []
res["max_trq"] = []
res["rew"] = []
for str_i in input_string:
    numbers = re.findall(pattern, str_i)
    numbers = [float(num) for num in numbers]
    res["min_trq"].append(min([numbers[0], numbers[1]]))
    res["max_trq"].append(max([numbers[0], numbers[1]]))
    res["rew"].append(numbers[2])
print(res)

plt.figure()
plt.plot(res["rew"], res["min_trq"])
plt.plot(res["rew"], res["max_trq"])
plt.title("Dependencies")
plt.xlabel("Reward")
plt.ylabel("Max/Min torque")
plt.ylim([20, 60])
plt.legend(["Min torque", "Max torque"])
plt.grid(True)
plt.show()