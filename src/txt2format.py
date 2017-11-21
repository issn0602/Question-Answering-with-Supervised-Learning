read_data = open("temp.txt")
import numpy as np
val = []
flag = 0
sum=0
for txt in read_data:
	if txt.find("?") == -1:
		sum = sum + len(txt.split())
	else:
		val.append(sum)
		sum=0

np_val = np.array(val)
print(np_val)
import matplotlib.pyplot as plt
plt.hist(np_val,range=[0,500])
plt.title("Histogram of Paragraph Length")
plt.xlabel("Paragraph Length")
plt.ylabel("Frequency")
plt.show()


import numpy as np
val = []
max_val = 0
loc = []
tot = 0
for txt in read_data:
	tot = tot+1
	if txt.find("?") >= 0:
		st = txt.index("?")
		ed = len(txt)-1
		if (ed-st) <= 200:
			temp = txt[0:(st-2)].count(" ")
			val.append(temp+1)
			
np_val = np.array(val)
import matplotlib.pyplot as plt
plt.hist(np_val,range=[0,40])
plt.title("Histogram of Question Length")
plt.xlabel("Question Length")
plt.ylabel("Frequency")
plt.show()

import numpy as np
val = []
quest = []
max_val = 0
loc = []
tot = 0
for txt in read_data:
	tot = tot+1
	if txt.find("?") >= 0:
		st = txt.index("?") + 2
		ed = len(txt)-1
		if (ed-st) <= 200:
			temp = txt[0:(st-2)].count(" ")
			quest.append(temp)
			temp = txt[st:ed].count(" ")
			val.append(temp)

np_val = np.array(val)
import matplotlib.pyplot as plt
plt.hist(np_val,range=[0,40])
plt.title("Histogram of Answer Length")
plt.xlabel("Answer Length")
plt.ylabel("Frequency")
plt.show()