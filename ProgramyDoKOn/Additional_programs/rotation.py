import numpy as np

def vectorLength(vector):
	suma = 0
	for i in range(0, len(vector)):
		suma += vector[i] ** 2
	return np.sqrt(suma)

def mulmul(a,b):
	return np.einsum ('ik,kj->ij',a,b)

def mulvec(a,b):
	return np.einsum('ik,k->i',a,b)

def mulmul3(a,b,c):
	return np.einsum('ik,kj,jm->im',a,b,c)

# MIXING ANGLE
sth13 = np.sqrt(0.0219)
sth23 = np.sqrt(0.304)
sth12 = np.sqrt(0.5)

cth13 = np.sqrt(1 - sth13 ** 2)
cth23 = np.sqrt(1 - sth23 ** 2)
cth12= np.sqrt(1 - sth12 ** 2)

STH13 = np.sqrt((1 - cth13) / 2)
STH23 = np.sqrt((1 - cth23) / 2)
STH12 = np.sqrt((1 - cth12) / 2)

CTH13 = np.sqrt((1 + cth13) / 2)
CTH23 = np.sqrt((1 + cth23) / 2)
CTH12 = np.sqrt((1 + cth12) / 2)

delta = 0

# The initial electron state

InitElectr = np.asarray([CTH12 * CTH13, STH12 * CTH13, STH13 * np.exp(-1j * delta)])
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

crossProductMatrix = np.asarray([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])

rotationMatrix = np.identity(3) + crossProductMatrix + mulmul(crossProductMatrix, crossProductMatrix) * 1/(1 + cosAlfa)

rotationMatrixNew = np.linalg.inv(rotationMatrix)

# The vectors in pentagram
 
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

# New vectors

v1New = mulvec(rotationMatrixNew, v1)
v2New = mulvec(rotationMatrixNew, v2)
v3New = mulvec(rotationMatrixNew, v3)
v4New = mulvec(rotationMatrixNew, v4)
v5New = mulvec(rotationMatrixNew, v5)

print(v1New)

#SIX DIMENSIONAL ROTATION

x = np.asarray([1,0,0,1,0,0])
y = np. asarray([1,2,1,1,2,1])

# THE GRAM-SCHMIDT

u = x/vectorLength(x)
v = y-np.inner(u,y)*u
v = v/vectorLength(v)

# Calculation of an angle

cosinus = np.inner(x,y)/(vectorLength(x)*vectorLength(y))
sinus = np.sqrt(1-cosinus * cosinus)

# Rotation matrix in 2 dimensions

Rotation2 = np.asarray([[cosinus,-sinus],[sinus, cosinus]])

# Projection

Q = np.identity(6) - np.outer(u,u) - np.outer(v,v)

lista = np. asarray([u,v])

Rot = Q + mulmul3(np.transpose(lista), Rotation2, lista)

wynik = mulvec(Rot,x)

# spytac sie

InitElectr = np.asarray([CTH12 * CTH13, STH12 * CTH13, STH13 * np.exp(-1j * delta)])
