# -*- coding: utf-8 -*-
#Реализация разрывного метода Галеркина, описанного в публикации
#Хайталиев И.Р., Шильников Е.В.
#Решение уравнений конвекции-диффузии локальным разрывным методом Галеркина //
#Препринты ИПМ им. М.В.Келдыша. 2023. № 17. 27 с. https://doi.org/10.20948/prepr-2023-17
#https://library.keldysh.ru/preprint.asp?id=2023-17
from scipy import integrate, special
import numpy as np
import matplotlib.pyplot as plt

#Задаем граничные условия
#В данной задаче граничные условия в x_L и x_R одинаковые
nu_l = lambda t: -np.exp(-a*t)*np.sin(c*t)
nu_r = lambda t: -np.exp(-a*t)*np.sin(c*t)

def evaluate_q(beta, N, M):
    q = np.zeros(M)
    #Значения полиномов Лежандра степени от 0 до N-1 в точке ksi = -1
    legendre_p = special.eval_legendre(range(N),-1)
    for k in range(M):
        if k == 0:
            #Плотность теплового потока в точке x_L
            q[k] = np.dot(beta[k,:], legendre_p)  
        elif k == M-1:
            #Поток в точке x_R
            q[k] = np.sum(beta[k-1,:])
        else:
            q[k] = 0.5*(np.dot(beta[k,:], legendre_p) + np.sum(beta[k-1,:]))
            
    return q  

def evaluate_u(t, alpha, N, M):
    u = np.zeros(M)
    #Значения полиномов Лежандра степени от 0 до N-1 в точке ksi = -1
    legendre_p = special.eval_legendre(range(N),-1)
    for k in range(M):
        if k == 0:
            #Температура в точке x_L
            u[k] = nu_l(t)     
        elif k == M-1:
            #Температура в точке x_R
            u[k] = nu_r(t)
        else:
            u[k] = 0.5*(np.dot(alpha[k,:], legendre_p) + np.sum(alpha[k-1,:])) 
    return u

def RK_TVD(alpha, beta, M, N, grid_x, t1, t2):
    alpha_temp = np.zeros((M-1, N))
    beta_temp = np.zeros((M-1, N))
    #Шаг по времени
    tau = t2-t1
    #Вычисляем значения плотности теплового потока на границах ячеек (t1)
    q = evaluate_q(beta, N, M)
     
    #Определяем значения коэффициентов alpha в момент времени t1+tau/2
    for k in range(M-1):
        for i in range(N):
            f = (2*i+1)*(q[k+1] - q[k]*(-1)**i - np.dot(beta[k,:], R[i,:]))/(grid_x[k+1]-grid_x[k])
            alpha_temp[k,i] = alpha[k,i] + tau*f/2  
    #Пересчитываем значения температуры на границах ячеек с новыми alpha (t1+tau/2)         
    u = evaluate_u(t1+tau/2, alpha_temp, N, M)
    #А затем значения коэффициентов beta (t1+tau/2)
    for k in range(M-1):
        for i in range(N):
            beta_temp[k, i] = -c*alpha_temp[k,i] +\
                a*(2*i+1)*(u[k+1] - u[k]*(-1)**i - np.dot(alpha_temp[k,:], R[i,:]))/(grid_x[k+1]-grid_x[k])
    #Вычисляем значения плотности теплового потока на границах ячеек  (t1+tau/2)        
    q = evaluate_q(beta_temp, N, M)
     
    #Значения коэффициентов alpha в момент времени t2
    for k in range(M-1):    
        for i in range(N):
            f = (2*i+1)*(q[k+1] - q[k]*(-1)**i -\
                         np.dot(beta_temp[k,:], R[i,:]))/(grid_x[k+1]-grid_x[k])
            alpha[k,i] = alpha[k,i] + tau*f
            
    #Пересчитываем значения температуры на границах ячеек (t2)        
    u = evaluate_u(t2, alpha, N, M)
    
    #Значения коэффициентов beta в момент времени t2    
    for k in range(M-1):
        for i in range(N):
            beta[k, i] = -c*alpha[k,i] +\
                a*(2*i+1)*(u[k+1] - u[k]*(-1)**i - np.dot(alpha[k,:], R[i,:]))/(grid_x[k+1]-grid_x[k])
            
    return alpha, beta  
            
#Задаем все необходимые параметры задачи для уравнения конвекции-диффузии
#a - коэффициент температуропроводности, c - скорость потока
#N - количество базисных функций, M - количество узлов (тогда ячеек M-1)
#x_L, x_R - границы рассматриваемой области (отрезок)
c = 1
a = 1
N = 2
M = 100

x_L = 0
x_R = 2*np.pi

#Строим сетку по пространству
grid_x = np.linspace(x_L, x_R, M) 
K = 4000
T = 2
grid_t = np.linspace(0, T, K)

#Массивы значений коэффициентов alpha, beta в момент времени t
#И массивы значений alpha, beta, используемых для вычисления коэффициентов
#в момент времени t+tau
alpha = np.zeros((M-1, N))
beta = np.zeros((M-1, N))

#Подынтегральные выражения для вычисления коэффициентов alpha, beta при t=0
F_alpha = lambda ksi, xk, hk, i: np.sin(xk + hk*(ksi+1)/2)*special.eval_legendre(i,ksi)

F_beta = lambda ksi, xk, hk, i: (-c*np.sin(xk + hk*(ksi+1)/2) +\
                                 a*np.cos(xk + hk*(ksi+1)))*special.eval_legendre(i,ksi)

#Определяем коэффициенты разложения решения в ряд в начальный момент времени
for k in range(M-1):
    for i in range(N):
        alpha[k,i] = (i+0.5)*integrate.quad(F_alpha, -1, 1, args=(grid_x[k], 
                                                      grid_x[k+1]-grid_x[k], i))[0]
        beta[k,i] = (i+0.5)*integrate.quad(F_beta, -1, 1, args=(grid_x[k], 
                                                grid_x[k+1]-grid_x[k], i))[0]

#R - матрица попарных скалярных произведений базисных функций и их производных 
#Корреляционная матрица
R = np.zeros((N, N))

for i in range(1, N):
    for j in range(N):
        if i == 1:
            R[i,j] = 2*int((i-1)==j)
        else:
            R[i,j] = 2*int((i-1)==j) + R[(i-2),j]

#Двухстадийный метод Рунге-Кутты 
for j in range(K-1):
    alpha, beta = RK_TVD(alpha, beta, M, N, grid_x, grid_t[j], grid_t[j+1]) 
              
#Находим значения температуры в узловых точках с помощью точного аналитического
#решения
U_analytical = np.exp(-a*T)*np.sin(grid_x-c*T)
#Находим значения температуры с помощью коэффициентов, полученных численно 
#Решение в узле k определяется суммированием по полиномам Лежандра степени от 0 до N-1
#с коэффициентами alpha[k,0:N]. 
#При этом значения полиномов Лежандра берем в точке ksi = -1 (локальная система координат)
U_numerical = np.zeros(M)
U_numerical[:M-1] = np.dot(alpha, special.eval_legendre(range(N),-1))
#Необходим дополнительный расчет температуры на правой границе.
#Так как количество ячеек I_k = M-1, количество точек пространственной сетки = M
#Поэтому проводим аналогичное суммирование по базисным функциям, взятым в точке ksi = 1
U_numerical[M-1] = np.dot(alpha[M-2,:], special.eval_legendre(range(N),1))

plt.plot(grid_x, U_numerical, c='m', label='analytical')
plt.plot(grid_x, U_analytical, 'gv', label='numerical')
plt.legend()
plt.show()
#Погрешность решения 
err = np.max(np.abs(U_numerical - U_analytical))
print('Погрешность численного решения', "%.5f" %err)