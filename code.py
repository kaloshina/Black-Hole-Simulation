import pygame, sys
import numpy as np

pygame.init()

WINDOW = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("Black Hole Sim")

WINDOW.fill((0, 0, 0))


# simulation is run in the equatorial plane, where the vertical angle theta = pi/2
method = 1 # method 1 is geodesic eq, method 2 uses conserved quantities and performs first integrals

# black hole properties (where G and c are 1)
M = 120.     # mass
J = 0.5*M**2.      # angular momentum
a = J/M
r_p = M + np.sqrt(M**2 - a**2) # outer event horizon

# setting up the black hole at the center of the screen
BHPos = np.array([960., 540.])

v = 1.  # speed of paths being traced out (not necessarily c)


class blackHole:
    def __init__(self, xpos, ypos):
        '''Black Hole only has a position attribute'''
        self.pos = np.array([xpos, ypos])
        self.draw()
    
    def draw(self):
        pygame.draw.circle(WINDOW, (255, 0, 0), self.pos, r_p)  # Draw a red circle for the black hole
        pygame.draw.circle(WINDOW, (0, 0, 0), self.pos, r_p-3)


class lightRay:
    def __init__(self, xpos, ypos, velAngle):
        '''Light Ray object'''
        # cartesian positions (absolute and relative to black hole)
        pos = np.array([xpos, ypos])
        self.posRel = pos - BHPos
        # velocity vector
        vel = [v * np.cos(velAngle), -v * np.sin(velAngle)]
        # polar coordinates
        self.r = np.sqrt(self.posRel[0]**2 + self.posRel[1]**2)
        self.phi = 0
        self.update_phi()
        # r' and phi'
        self.dr = (self.posRel[0]*vel[0]+self.posRel[1]*vel[1])/self.r
        self.dphi = (self.posRel[0]*vel[1]-self.posRel[1]*vel[0])/self.r**2

        # conserved quantity E
        Delta = self.r*(self.r - 2.*M) + a**2
        r = self.r
        A = 2.*M/r-1
        B = -4.*M*a*self.dphi/r
        C = r**2*self.dr**2/Delta + (r**2 + a**2 + 2.*M*a**2/r)*self.dphi**2
        dt = (-B + np.sqrt(B**2-4.*A*C))/(2.*A)
        self.E = (1-2.*M/self.r)*dt + 2.*M*a*self.dphi/self.r

        self.alive = (self.r >= r_p)
        if self.alive:
            self.draw()

    def update(self, dl):
        if self.alive:
            # conduct RK4 on r' and phi'
            if method == 1:
                k1 = self.d2r(self.dr)
                k2 = self.d2r(self.dr+dl*k1/2)
                k3 = self.d2r(self.dr+dl*k2/2)
                k4 = self.d2r(self.dr+dl*k3)
                self.dr += dl/6*(k1+2*k2+2*k3+k4)
                k1 = self.d2phi(self.dphi)
                k2 = self.d2phi(self.dphi+dl*k1/2)
                k3 = self.d2phi(self.dphi+dl*k2/2)
                k4 = self.d2phi(self.dphi+dl*k3)
                self.dphi += dl/6*(k1+2*k2+2*k3+k4)
                # update r and phi positions
                self.r += self.dr*dl
                self.phi += self.dphi*dl
            elif method == 2:
                pass
            # convert position to cartesian and update location
            self.posRel = np.array([self.r*np.cos(self.phi), self.r*np.sin(self.phi)])
            # check if the light has crossed the event horizon
            self.alive = (self.r > r_p+1)
            if self.alive:
                self.draw()

    def d2r(self, dr):
        # radial acceleration
        '''if a == 0: return -M*(self.r-2.*M)/(self.r**3)*(self.E/(1.-2.*M/self.r))**2 \
                                + M/(self.r*(self.r-2.*M))*dr**2 \
                                + (self.r-2.*M)*self.dphi**2'''
        Delta = self.r*(self.r - 2.*M) + a**2
        dt = (self.E*self.r - 2.*M*a*self.dphi)/(self.r-2.*M)
        return -M*Delta/self.r**4.*dt**2 \
                + 2.*M*a*Delta/self.r**4*dt*self.dphi \
                - (a**2-M*self.r)/(self.r*Delta)*dr**2 \
                + Delta*(self.r**3-M*a**2)/self.r**4*self.dphi**2
        
        
    def d2phi(self, dphi):
        # angular acceleration (derived using geodesic equation)
        '''if a == 0: return -2.*self.dr*dphi/self.r'''
        Delta = self.r*(self.r - 2.*M) + a**2
        dt = (self.E*self.r - 2.*M*a*self.dphi)/(self.r-2.*M)
        return -2.*self.dr/(self.r**2*Delta)*(M*a*dt + (self.r**3-2.*M*self.r**2-M*a**2)*self.dphi)
    
    def update_phi(self):
        # properly sets up phi due to the range limitations of arctan
        if (self.posRel[0] != 0):
            if (self.posRel[1] != 0):
                self.phi = np.arctan(self.posRel[1]/self.posRel[0])
                if (self.phi < 0 and self.posRel[0] < 0): self.phi += np.pi
                elif (self.phi > 0 and self.posRel[0] < 0): self.phi -= np.pi
            else:
                if (self.posRel[0] < 0): self.phi = np.pi
                else: self.phi = 0
        else:
            if (self.posRel[1] > 0): self.phi = np.pi/2
            elif (self.posRel[1] == 0): self.phi = 0
            else: self.phi = -np.pi/2        
    
    def draw(self):
        n=0.1
        pygame.draw.circle(WINDOW, self.wavelength_to_rgb(580/np.sqrt(1-n*r_s/self.r)), self.posRel+BHPos, 1)

    def wavelength_to_rgb(self, wavelength):
        if 380 <= wavelength < 440: R = -(wavelength - 440) / (440 - 380); G = 0.0; B = 1.0
        elif 440 <= wavelength < 490: R = 0.0; G = (wavelength - 440) / (490 - 440); B = 1.0
        elif 490 <= wavelength < 510: R = 0.0; G = 1.0; B = -(wavelength - 510) / (510 - 490)
        elif 510 <= wavelength < 580: R = (wavelength - 510) / (580 - 510); G = 1.0; B = 0.0
        elif 580 <= wavelength < 645: R = 1.0; G = -(wavelength - 645) / (645 - 580); B = 0.0
        elif 645 <= wavelength <= 750: R = 1.0; G = 0.0; B = 0.0
        else: R = G = B = 0.0
        R = int(R * 255); G = int(G * 255); B = int(B * 255) 
        return (R, G, B)


# setting up the black hole
BH1 = blackHole(BHPos[0], BHPos[1])

# setting up the light rays
n = 216
LRList = [None]*n
for i in range(n):
    #LRList[i] = lightRay(100, i*10, 0)
    LRList[i] = lightRay(0, -540.+i*10., 0)
#LRList[n-1] = lightRay(BHPos[0], BHPos[1]-3.*M, 0)

# running each event sequentially
while True:
    # update each light ray
    for LR in LRList:
        LR.update(1)
    # check if window is closed  
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()

