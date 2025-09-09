import pygame, sys
import numpy as np

pygame.init()

WINDOW = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("Black Hole Sim")

WINDOW.fill((0, 0, 0))

# black hole properties (where G and c are 1)
M = 50
J = 0 
a = J/M
r_s = 2*M
r_p = r_s/2 + np.sqrt((r_s/2)**2 - a**2) # outer event horizon

BHPos = np.array([960, 540])

v = 5000.  # speed of paths being traced out (not necessarily c)



class blackHole:
    def __init__(self, xpos, ypos):
        self.pos = np.array([xpos, ypos])
        self.draw()
    
    def draw(self):
        pygame.draw.circle(WINDOW, (255, 0, 0), self.pos, r_s)  # Draw a red circle for the black hole
        pygame.draw.circle(WINDOW, (0, 0, 0), self.pos, r_s-2)


class lightRay:
    def __init__(self, xpos, ypos, velAngle):
        # cartesian position and velocity
        pos = np.array([xpos, ypos])
        self.posRel = pos - BHPos
        vel = [v * np.cos(velAngle), -v * np.sin(velAngle)]
        # polar position and velocity
        self.r = np.sqrt(self.posRel[0]**2 + self.posRel[1]**2)
        self.phi = 0
        self.update_phi()
        # r' and phi'
        self.dr = (self.posRel[0]*vel[0]+self.posRel[1]*vel[1])/self.r
        self.dphi = (self.posRel[0]*vel[1]-self.posRel[1]*vel[0])/self.r**2

        # Kerr metric variables   
        self.Sigma = self.r**2
        self.Delta = self.r**2 - r_s*self.r + a**2

        # conserved quantity E
        if a == 0:
            self.E = np.sqrt(self.dr**2 + (self.r**2-self.r*r_s)*self.dphi**2)
        else:
            A = r_s*self.r/self.Sigma-1
            B = -2*r_s*self.r*a*self.dphi/self.Sigma
            C = self.Sigma/self.Delta*self.dr**2+(self.r**2+a**2+r_s*self.r*a**2/self.Sigma)*self.dphi**2
            D = (-B+np.sqrt(B**2-4*A*C))/(2*A)
            self.E = D*(r_s-self.r)/(r_s*a*self.dphi)

        self.alive = (self.r >= r_p)
        if self.alive:
            self.draw()

    def update(self, dl):
        if self.alive:

            # conduct RK4 on r', phi', r and phi
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
            # convert position to cartesian and update location
            self.posRel = np.array([self.r*np.cos(self.phi), self.r*np.sin(self.phi)])
            # check if the light has crossed the event horizon
            self.alive = (self.r >= r_p)
            if self.alive:
                self.draw()

    def d2r(self, dr):
        if a == 0: return (r_s*(r_s-self.r)/(2.*self.r**3))*(self.E/(1.-r_s/self.r))**2 \
                                - r_s/(2.*self.r*(self.r-r_s))*dr**2 \
                                + (self.r-r_s)*self.dphi**2
        else:
            dt = (self.E*self.r - r_s*a*self.dphi)/(self.r-r_s)
            return M/self.r**2*(1-2*M/self.r+a**2/self.r**2)*dt**2 \
                    - 2*M*a/self.r**2*dt*self.dphi \
                    - (self.r**2-self.r*M-M)/(self.r*self.Delta)*dr**2 \
                    + (self.r-2*M+a**2/self.r)*self.dphi**2
    
    def d2phi(self, dphi):
        if a == 0: return -2.*self.dr*dphi/self.r
        else:
            dt = (self.E*self.r - r_s*a*self.dphi)/(self.r-r_s)
            return  -2*M*(self.r**2-a**2)/(self.r**2*self.Delta)*dt*self.dr \
                    + 2*M/self.r**2*dt*dphi \
                    - a/(self.r*self.Delta)*self.dr**2 \
                    + M/self.r**2*dphi**2

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
        n=0.2
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
n = 108
LRList = [None]*n
for i in range(n):
    LRList[i] = lightRay(0, -500+i*10, -0.7)
    



# running each event sequentially
while True:
    # update each light ray
    for LR in LRList:
        LR.update(0.0001)
    # check if window is closed  
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()


