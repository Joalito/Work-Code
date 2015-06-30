"""
AB needs super low cfl (<.1), RK3 can go as high as around .7, RK4, .8
"""
import numpy as np
from numpy import roll
import matplotlib.pyplot as plt
from copy import copy
import matplotlib.animation as animation

dx = 1.
nx  = 150
cfl = .3
phys_len   = dx * nx
dudt = np.array([])
dudtm1 = np.array([])
dudtm2 = np.array([])

def WENO(thing):
    """
    I have to split this up. It has to return a value depending on if
    the original value was positive or negative. It is a small waste
    but I can just calculate the entire thing both the positive way and
    the negative way, then select out the values that were originally
    positive or negative and then put them into the final array
    """
    # determine the indices where positive
    positive_indices = []
    for i in range(len(thing)):
        if thing[i] >= 0:
            positive_indices.append(i)
    # determine the indices where negative
    negative_indices = []
    for i in range(len(thing)):
        if thing[i] < 0:
            negative_indices.append(i)
    # calculate positive

    um2 = roll(thing, 2)
    um1 = roll(thing, 1)
    up1 = roll(thing, -1)
    up2 = roll(thing, -2)

    u0 = (1./3.)*um2 - (7./6.)*um1 + (11./6.)*thing
    u1 = -(1./6.)*um1 + (5./6.)*thing + (1./3.)*up1
    u2 = (1./3.)*thing + (5./6.)*up1 - (1./6.)*up2

    # nonlinear weights

    epsilon = 10**-6

    beta0 = (13./12.)*(um2 - 2*um1 + thing)**2 + .25*(um2 - 4*um1 + 3*thing)**2
    beta1 = (13./12.)*(um1 - 2*thing + up1)**2 + .25*(um1 - up1)**2
    beta2 = (13./12.)*(thing - 2*up1 + up2)**2 + .25*(3*thing - 4*up1 + up2)**2

    beta = [beta0, beta1, beta2]
    gamma = [.1, .6, .3]

    omegatilde = [0, 0, 0]
    for i in range(3):
        omegatilde[i] = gamma[i] / (epsilon + beta[i])**2

    omega = [0, 0, 0]
    for i in range(3):
        omega[i] = omegatilde[i] / sum(omegatilde)

    u_hlf = omega[0]*u0 + omega[1]*u1 + omega[2]*u2

    flux = (u_hlf)**2
    duudx = - (flux - roll(flux, 1)) / dx

    # calculate negative
    um1 = np.roll(thing, 1)
    up1 = np.roll(thing, -1)
    up2 = np.roll(thing, -2)
    up3 = np.roll(thing, -3)

    u0 = (1./3.)*up3 - (7./6.)*up2 + (11./6.)*up1
    u1 = -(1./6.)*up2 + (5./6.)*up1 + (1./3.)*thing
    u2 = (1./3.)*up1 + (5./6.)*thing - (1./6.)*um1

    # nonlinear weights

    epsilon = 10**-6

    beta0 = (13./12.)*(up3 - 2*up2 + up1)**2 + .25*(up3 - 4*up2 + 3*up1)**2
    beta1 = (13./12.)*(up2 - 2*up1 + thing)**2 + .25*(up2 - thing)**2
    beta2 = (13./12.)*(up1 - 2*thing + um1)**2 + .25*(3*up1 - 4*thing + um1)**2

    beta = [beta0, beta1, beta2]
    gamma = [.1, .6, .3]
    omegatilde = [0, 0, 0]
    for i in range(3):
        omegatilde[i] = gamma[i] / (epsilon + beta[i])**2

    omega = [0, 0, 0]

    for i in range(3):
        omega[i] = omegatilde[i] / sum(omegatilde)

    u_hlfneg = omega[0]*u0 + omega[1]*u1 + omega[2]*u2
    fluxneg = (u_hlfneg)**2
    duudxneg = - (fluxneg - roll(fluxneg, 1)) / dx

    # extract values where was positive
    positive_values = np.extract(thing>=0, duudx)
    # negative
    negative_values = np.extract(thing<0, duudxneg)

    # combine into new list
    positive = np.zeros(nx)
    np.put(positive, positive_indices, positive_values)
    negative = np.zeros(nx)
    np.put(negative, negative_indices, negative_values)
    dudt = np.zeros(nx)
    np.put(dudt, positive_indices, positive_values)
    np.put(dudt, negative_indices, negative_values)
    return dudt

# define delta function
delta = []
for _ in range(nx/3):
    delta.append(0)
for _ in range(nx/3):
    delta.append(1)
for _ in range(nx/3):
    delta.append(0)

class ValuesOnGrid:

    def __init__(self, X=np.array([]), currentstate=np.array([]), time_elapsed=0):

        self.X          = np.asarray([i*dx for i in range(nx)])
        self.currentstate = np.sin(2*np.pi*self.X / phys_len)**2 + 2
        #self.currentstate = np.asarray(delta)
        self.time_elapsed = time_elapsed
        self.dt         = (cfl * dx) / np.amax(self.currentstate)


    def stepRK3(self):

        # 3rd-Order Runge-Kutta time integration scheme:
            # temp1 = u + dt*L(u, dx)
            # temp2 = .75*u + .25*temp1 + .25*dt*L(temp1, dx)
            # u = (1./3.)*u + (2./3.)*temp2 + (2./3.)*dt*L(temp2, dx)

        temp1 = self.currentstate + self.dt*WENO(self.currentstate)
        temp2 = .75*self.currentstate + .25*temp1 + .25*self.dt*WENO(temp1)

        self.currentstate = (1./3.)*self.currentstate + (2./3.)*temp2 + (2./3.)*self.dt*WENO(temp2)
        self.time_elapsed += self.dt


    def stepRK4(self):

        # 4th-Order Runge-Kutte Time Integration scheme
            # u1 = u + .5*dt*L(u)
            # u2 = u + .5*dt*L(u1)
            # u3 = u + dt*L(u2)
            # u = (1./3.)*(-u + u1 + 2*u2 + u3) + (1./6.)*dt*L(u3)

        temp1 = self.currentstate + .5*self.dt*WENO(self.currentstate)
        temp2 = self.currentstate + .5*self.dt*WENO(temp1)
        temp3 = self.currentstate + self.dt*WENO(temp2)

        self.currentstate = (1./3.)*(-self.currentstate + temp1 + 2*temp2 + temp3)
        + (1./6.)*self.dt*WENO(temp3)

        self.time_elapsed += self.dt


    def setupAB(self):
        global dudt
        global dudtm1

        dudt = WENO(self.currentstate)
        dudtm1 = copy(dudt)

        #      u          +=      dt*dudt
        self.currentstate += self.dt*dudt

        dudt = WENO(self.currentstate)

        #       u         =         u         + (1.5*dudt - .5*dudtm1) *      dt
        self.currentstate = self.currentstate + (1.5*dudt - .5*dudtm1) * self.dt



    def stepAB(self):
        global dudt
        global dudtm1
        global dudtm2

        dudtm2 = copy(dudtm1)
        dudtm1 = copy(dudt)
        dudt = WENO(self.currentstate)

        #       u         =     u             + ((23./12.)*dudt - (4./3.)*dudtm1 + (5./12.)*dudtm2) *      dt
        self.currentstate = self.currentstate + ((23./12.)*dudt - (4./3.)*dudtm1 + (5./12.)*dudtm2) * self.dt
        self.time_elapsed += self.dt

    def energy(self):
        K = .5 * np.sum((self.currentstate)**2)
        return K




timeIntegration = 'RK3'


u = ValuesOnGrid()

if timeIntegration == 'AB':
    u.setupAB()



# generate figure and put all the text in place
fig = plt.figure(1)
ax  = plt.axes(xlim=(0, (nx + 5)), ylim=((np.amin(u.currentstate)-.5), (np.amax(u.currentstate)+.5)))
ax.grid()
line, = ax.plot([], [])
plt.title("%s WENO nx = %d" % (timeIntegration, nx - 1))
ax.text(10, (np.amax(u.currentstate)+.4), 'cfl = %.2f' % cfl)
time_text   = ax.text(10, (np.amax(u.currentstate) + .25), '')
energy_text = ax.text(10, (np.amax(u.currentstate) + .10), '')




# called between frames to clear the figure
def init():
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text


# animation function.  This is called sequentially
def animate(i):
    if timeIntegration == 'AB':
        u.stepAB()
    elif timeIntegration == 'RK3':
        u.stepRK3()
    elif timeIntegration == 'RK4':
        u.stepRK4()
    line.set_data(u.X, u.currentstate)
    time_text.set_text('time = %.1f' % u.time_elapsed)
    energy_text.set_text('energy = %.3f J' % u.energy())
    return line, time_text, energy_text

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=10, interval=20, blit=True)
