import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import time
import matplotlib.pyplot as plt
import matplotlib

#				                                 SOME BASIC FUNCTIONS									

def mulvec(a,b):
	return np.einsum('ik,k->i',a,b)

def mulmul(a,b):
	return np.einsum ('ik,kj->ij',a,b)

def mulmul3(a,b,c):
	return np.einsum('ik,kj,jm->im',a,b,c)

# Inner Product for Complex Vectors (numpy provides only for the real ones)

def vectorLength(vector):
	suma = 0
	for i in range(0, len(vector)):
		suma += vector[i] ** 2
	return np.sqrt(suma)

def dotpr(v1,v2):
	v2 = np.conjugate(v2)
	return np.inner(v1,v2)

def outerpr(v1,v2):
	v1 = np.conjugate(v1)
	return np.outer(v1,v2)

# Verification of the positivity of the map

def isPositive (CMatrix):
	for i in range(0,9):
		for j in range(0,9):
			if (abs(CMatrix[i][j]) > 0.5 * (CMatrix[i][i] + CMatrix[j][j])):
				return False
	return True

#                                                      SU(3) GENERATORS

F0 = np.identity(3)
F1 = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
F2 = np.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
F3 = np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
F4 = np.asarray([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
F5 = np.asarray([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
F6 = np.asarray([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
F7 = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
F8 = 1 / (np.sqrt(3)) * np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -2]])

# Map from a matrix to a vector (Gellmann representation)

def mapMatrixToVector(matrix):
	skl0 = 1.0/3.0 * np.trace(mulmul(F0, matrix))
	skl1 = 0.5 * np.trace(mulmul(F1, matrix))
	skl2 = 0.5 * np.trace(mulmul(F2, matrix))
	skl3 = 0.5 * np.trace(mulmul(F3, matrix))
	skl4 = 0.5 * np.trace(mulmul(F4, matrix))
	skl5 = 0.5 * np.trace(mulmul(F5, matrix))
	skl6 = 0.5 * np.trace(mulmul(F6, matrix))
	skl7 = 0.5 * np.trace(mulmul(F7, matrix))
	skl8 = 0.5 * np.trace(mulmul(F8, matrix))

	vector = np.asarray([skl0, skl1, skl2, skl3, skl4, skl5, skl6, skl7, skl8])
	return vector

#                                                     PARAMETERS

# Mixing angles (sth- sinus 2*theta, cth- cosinus 2*theta)

sth13 = np.sqrt(0.0219)
sth23 = np.sqrt(0.304)
sth12 = np.sqrt(0.5)

cth13 = np.sqrt(1 - sth13 ** 2)
cth23 = np.sqrt(1 - sth23 ** 2)
cth12 = np.sqrt(1 - sth12 ** 2)

STH13 = np.sqrt((1 - cth13) / 2)
STH23 = np.sqrt((1 - cth23) / 2)
STH12 = np.sqrt((1 - cth12) / 2)

CTH13 = np.sqrt((1 + cth13) / 2)
CTH23 = np.sqrt((1 + cth23) / 2)
CTH12 = np.sqrt((1 + cth12) / 2)

angle1 = STH12 * STH12
angle2 = STH13 * STH13
angle3 = STH23 * STH23
angle4 = CTH12 * CTH12
angle5 = CTH13 * CTH13
angle6 = CTH23 * CTH23

# The mass squares

m21 = 7.53 * 10  # w milielektronowoltach
m32 = 2.44 * 10 ** (3)
m31 = m21 + m32

# The CP violation phase

a = 0

# The remaining parameters

En = 10 ** (11) # energy of the neutrino
Enodwr = 1.0/(En)
VCC = 0.001 # charge-current potential


#                                                    DISSIPATIVE PART

#                                     Dissipative part - read from file 

stala = 0.1

C00 = C11 = C22 = C33 = C44 = C55 = C66 = C77 = C88 = C99 = stala

# off-diagonal part

C12 = C21 = stala
C13 = C31 = stala
C08 = C80 = stala

C01 = C02 = C03 = C04 = C05 = C06 = C07  = stala
C10 = C14 = C15 = C16 = C17 = C18 = stala
C20 = C23 = C24 = C25 = C26 = C27 = C28 = stala
C30 = C32 = C34 = C35 = C36 = C37 = C38 = stala
C40 = C41 = C42 = C43 = C45 = C46 = C47 = C48 = stala
C50 = C51 = C52 = C53 = C54 = C56 = C57 = C58 = stala
C60 = C61 = C62 = C63 = C64 = C65 = C67 = C68 = stala
C70 = C71 = C72 = C73 = C74 = C75 = C76 = C78 = stala
C81 = C82 = C83 = C84 = C85 = C86 = C87 = stala

CMatrix = [[C00,C01,C02,C03, C04, C05, C06, C07, C08],\
[C10, C11, C12, C13, C14, C15, C16, C17, C18],\
[C20, C21 ,C22 ,C23, C24, C25, C26, C27, C28], \
[C30, C31, C32, C33, C34, C35, C36, C37, C38],\
[C40, C41, C42, C43, C44, C45, C46, C47, C48],\
[C50, C51, C52, C53, C54, C55, C56, C57, C58],\
[C60, C61 ,C62 ,C63, C64, C65, C66, C67, C68],\
[C70, C71, C72, C73, C74, C75, C76, C77, C78],\
[C80, C81, C82, C83, C84, C85, C86, C87, C88]]


#                                                        OBSERVABLES
# Parameters
 
BigCos = np.cos(2 * np.pi / 5)
LittleCos = np.cos(np.pi / 5)
BigSin = np.sin(2 * np.pi / 5)
LittleSin = np.sin(np.pi / 5)

# Normalization constant

N = float(1) / float(np.sqrt(1 + LittleCos))

# Vectors

v1 = N * np.asarray([np.sqrt(LittleCos),1,0])
v2 = N * np.asarray([np.sqrt(LittleCos),-LittleCos,LittleSin])
v3 = N * np.asarray([np.sqrt(LittleCos),BigCos,-BigSin])
v4 = N * np.asarray([np.sqrt(LittleCos),BigCos, BigSin])
v5 = N * np.asarray([np.sqrt(LittleCos),-LittleCos,-LittleSin])

# Rotation
InitElectr = np.asarray([CTH12 * CTH13, STH12 * CTH13, STH13 * np.exp(-1j * a)])
VectorSimple = np. asarray([1,0,0])

# The cross product of vectors - the value is the sin of the rotation angle

crossProduct = np.cross(InitElectr, VectorSimple)

# The coordinates of the cross product

w1 = crossProduct[0]
w2 = crossProduct[1]
w3 = crossProduct[2]

sinAlfa = vectorLength(crossProduct)

# The dot product of vectors - the value is the cos of the rotation angle

cosAlfa = np.dot(InitElectr, VectorSimple)

# Creation of the Rotation Matrix 

crossProductMatrix = np.asarray([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
rotationMatrix = np.identity(3) + crossProductMatrix + mulmul(crossProductMatrix, crossProductMatrix) * 1/(1 + cosAlfa)
rotationMatrixNew = np.linalg.inv(rotationMatrix)

# New vectors

v1New = mulvec(rotationMatrixNew, v1)
v2New = mulvec(rotationMatrixNew, v2)
v3New = mulvec(rotationMatrixNew, v3)
v4New = mulvec(rotationMatrixNew, v4)
v5New = mulvec(rotationMatrixNew, v5)

# Observables

A1 = -2 * np.outer(v1New,v1New) + np.identity(3)
A2 = -2 * np.outer(v2New,v2New) + np.identity(3)
A3 = -2 * np.outer(v3New,v3New) + np.identity(3)
A4 = -2 * np.outer(v4New,v4New) + np.identity(3)
A5 = -2 * np.outer(v5New,v5New) + np.identity(3)

File1 = open("MathLinH.txt","r")
HMain = File1.read()
HMain = eval(HMain)

#                                   NONUNITARY EVOLUTION

# Initial states:

InitElectr = np.asarray([CTH12 * CTH13, STH12 * CTH13, STH13 * np.exp(-1j * a)])
InitElectrMixed = outerpr(InitElectr,InitElectr)
InitElectr = mapMatrixToVector(InitElectrMixed)

InitMuon = np.asarray([-STH12 * CTH23 - CTH12 * STH23 * STH13 * np.exp(-1j*a), \
	CTH12 * CTH23 - STH12 * STH23 * STH13 * np.exp(-1j*a), STH23 * CTH13 ])
InitMuonMixed = outerpr(InitMuon, InitMuon)
InitMuon = mapMatrixToVector(InitMuonMixed)

InitTaon = np.asarray([STH12 * STH23 - CTH12 * CTH23 * STH13 * np.exp(-1j*a), \
	-CTH12 * STH23 - STH12 * CTH23 * STH13 * np.exp(-1j*a), CTH23 * CTH13])
InitTaonMixed = outerpr(InitTaon, InitTaon)
InitTaon = mapMatrixToVector(InitTaonMixed)

Evolution = []
h = 0.01 # time interval
start = time.time()

for j in range(21):
# diagonal part
	F = open("MathLinDiss.txt","r")
	HDISS = F.read()
	HDISS = eval(HDISS)
	HDISS = np.transpose(HDISS)
	Heff = np.asarray(HDISS) + np.asarray(HMain)
	Czas = []
	Evolution1 = []
	for i in range(0,200):
		t = h * i
		Czas.append(t)
		wykladnik = np.asarray(Heff) * t
		NonUnitary = SLA.expm(wykladnik)
		NewState = mulvec(NonUnitary,InitElectr)
		Density = F0 * NewState[0] + F1 * NewState[1] + F2 * NewState[2] + F3 * NewState[3] + F4 * NewState[4] +\
		F5 * NewState[5] + F6 * NewState[6] + F7 * NewState[7] + F8 * NewState[8]
		ExpW1 = np.trace(mulmul3(Density,A1,A2))
		ExpW2 = np.trace(mulmul3(Density,A2,A3))
		ExpW3 = np.trace(mulmul3(Density,A3,A4))
		ExpW4 = np.trace(mulmul3(Density,A4,A5))
		ExpW5 = np.trace(mulmul3(Density,A5,A1))
		suma = ExpW1 + ExpW2 + ExpW3 + ExpW4 + ExpW5
		Evolution1.append(suma)
	Evolution.append(Evolution1)
 
stop = time.time()
czas = stop-start
print("czas = {}".format(czas))

plt.plot(Czas, Evolution[0], linewidth = 2, label = "$c_{ij}=0$")
plt.plot(Czas,Evolution[5],'y-',linewidth = 2,label = "$c_{ij}=0.5$")
plt.plot(Czas,Evolution[10],'r-',linewidth = 2,label="$c_{ij}=1$")
plt.plot(Czas,Evolution[15],'m-',linewidth=2,label="$c_{ij}=1.5$")
plt.plot(Czas,Evolution[20],'g-',linewidth=2,label="$c_{ij}=2$")
#plt.axhline(y = -3, color = 'black', linewidth = 3, hold = None)
ax1 = plt.subplot()
ax1.set_xlabel('Time');
ax1.set_ylabel('$\Delta$');
plt.legend()
plt.show()		

								