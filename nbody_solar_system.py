####Import modules####
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

####Constants/user inputs####
Gconstant = 6.67408e-11
AU = 1.495978707e11
TerraMass = 5.9742e24
maximumTime = 365.25*24*60*60*166
dt = 60*60*60
tArray = np.arange(dt,maximumTime+dt,dt)


####A generic body class that any body in the system will use####
class body:
    ####Inital values####
    def __init__(self, name, colour, mass, x, y, v_x, v_y):
        self.name = name
        self.colour = colour
        self.mass = mass
        self.x = x
        self.y = y
        self.tempX = x
        self.tempY = y
        self.v_x = v_x
        self.v_y = v_y


    #Note: We do the RK in steps because at each partial step the accel is calcaulted which depends on the position of all bodies
    #Therefore we must update each step for all bodies at once and not just do a single one in one go. Hence the split up functions

    ####Runge-kutta functions separately by step 1-4 and final:####
    def RK1(self):
        #Calculate current acceleration in x and y due to forces from all bodies
        acellX = 0
        acellY = 0
        for currentBody in allBodies:   #Add the acceleration from each body in simulation
            if(currentBody.name!=self.name):
                xDistance = currentBody.x - self.x
                yDistance = currentBody.y - self.y
                distanceBetween = math.sqrt(xDistance**2+yDistance**2)  #Actual distance between current and test bodies
                acellX += Gconstant*currentBody.mass*(xDistance)/distanceBetween**3
                acellY += Gconstant*currentBody.mass*(yDistance)/distanceBetween**3

        #Actual Runge-Kutta calculations
        self.k1_vx = self.v_x
        self.k1_vy = self.v_y
        self.k1_ax = acellX
        self.k1_ay = acellY

        #We update this bodies position in a temporary variable now @(dt/2) ahead as it will be referenced in RK2 by other bodies
        self.tempX = self.x + (dt/2)*self.k1_vx
        self.tempY = self.y + (dt/2)*self.k1_vy


        
    #Mostly same as RK1 bar acceleration calculation
    def RK2(self):
        #Calculate current acceleration in x and y due to forces from all bodies
        acellX = 0
        acellY = 0

        #At this stage the system is a state (dt/2) from the inital. The tempX and tempY values calculated
        #before are the positions for this state so we use them to calculate the current distances instead of self.x etc
        for currentBody in allBodies:
            if(currentBody.name!=self.name):
                xDistance = currentBody.tempX-self.tempX
                yDistance = currentBody.tempY-self.tempY
                distanceBetween = math.sqrt(xDistance**2+yDistance**2)
                acellX += Gconstant*currentBody.mass*(xDistance)/distanceBetween**3
                acellY += Gconstant*currentBody.mass*(yDistance)/distanceBetween**3

        #Actual Runge-Kutta calculations
        self.k2_vx = self.v_x + (dt/2)*self.k1_ax
        self.k2_vy = self.v_y + (dt/2)*self.k1_ay
        self.k2_ax = acellX
        self.k2_ay = acellY

        #We update this bodies position in a temporary variable now @(dt/2) ahead as it will be referenced in RK2 by other bodies
        self.tempX = self.x + (dt/2)*self.k2_vx
        self.tempY = self.y + (dt/2)*self.k2_vy



    #(Same format as RK2)
    def RK3(self):
        acellX = 0
        acellY = 0

        for currentBody in allBodies:
            if(currentBody.name!=self.name):
                xDistance = currentBody.tempX-self.tempX
                yDistance = currentBody.tempY-self.tempY
                distanceBetween = math.sqrt(xDistance**2+yDistance**2)
                acellX += Gconstant*currentBody.mass*(xDistance)/distanceBetween**3
                acellY += Gconstant*currentBody.mass*(yDistance)/distanceBetween**3

        #Actual Runge-Kutta calculations
        self.k3_vx = self.v_x + (dt/2)*self.k2_ax
        self.k3_vy = self.v_y + (dt/2)*self.k2_ay
        self.k3_ax = acellX
        self.k3_ay = acellY

        #Update temp position for next step
        self.tempX = self.x + (dt)*self.k3_vx
        self.tempY = self.y + (dt)*self.k3_vy



    #(Same format as RK2 but doesn't need to update temp variables for RKFinal)
    def RK4(self):   
        acellX = 0
        acellY = 0

        for currentBody in allBodies:
            if(currentBody.name!=self.name):
                xDistance = currentBody.tempX-self.tempX
                yDistance = currentBody.tempY-self.tempY
                distanceBetween = math.sqrt(xDistance**2+yDistance**2)
                acellX += Gconstant*currentBody.mass*(xDistance)/distanceBetween**3
                acellY += Gconstant*currentBody.mass*(yDistance)/distanceBetween**3

        #Actual Runge-Kutta calculations
        self.k4_vx = self.v_x + (dt)*self.k3_ax
        self.k4_vy = self.v_y + (dt)*self.k3_ay
        self.k4_ax = acellX
        self.k4_ay = acellY


    #The final Runge-Kutta step actually changing the bodies position and velocity
    def RKfinal(self):
        self.x = self.x + (dt/6)*(self.k1_vx+2*self.k2_vx+2*self.k3_vx+self.k4_vx)
        self.y = self.y + (dt/6)*(self.k1_vy+2*self.k2_vy+2*self.k3_vy+self.k4_vy)

        self.v_x = self.v_x + (dt/6)*(self.k1_ax+2*self.k2_ax+2*self.k3_ax+self.k4_ax)
        self.v_y = self.v_y + (dt/6)*(self.k1_ay+2*self.k2_ay+2*self.k3_ay+self.k4_ay)
   
        
    def totalMass(self):
        total = 0
        for singleBody in allBodies:
            total+= singleBody.mass
        print(total)





####Create all bodies and add them to a list####
#In the format: name, mass, x, y, v_x, v_y):        
allBodies = [
body("Sun", "#FFD800", 1.98892e30,0,0,0,-15.9918860487),
body("Mercury", "#8A7972", (0.0553*TerraMass),(0.39*AU),0,0,47699.00854),
body("Venus", "#DA7621", (0.815*TerraMass),(0.723*AU),0,0,35032.61101),
body("Earth", "#0026FF", (1*TerraMass),(1*AU),0,0,29788.02128),
body("Mars", "#C6482F", (0.107*TerraMass),(1.524*AU),0,0,24129.54718),
body("Jupiter","#9E5729",(317.83*TerraMass) ,(5.203*AU),0,0,13059.14496),
body("Saturn", "#C2B098", (95.162*TerraMass),(9.539*AU),0,0,9644.733483),
body("Uranus", "#006D98", (14.536*TerraMass),(19.18*AU),0,0,6801.698141),
body("Neptune", "#648ADF", (17.147*TerraMass),(30.06*AU),0,0,5433.093354)
]



#Contains arrays for each body with all their position cords over time
historyArray = []

#Prepare a blank history array that contains an array of position at each timestep for each body
for singleBody in allBodies:
    historyArray.append([])


#Add the starting locations to each array
bodyIndex = 0        
for singleBody in allBodies:
    historyArray[bodyIndex].append([singleBody.x,singleBody.y,singleBody.v_x,singleBody.v_y,0])
    bodyIndex+=1



####Run the simulation itself####
for time in tArray: #For each discrete timestep compute the system's state
    #RK1
    for singleBody in allBodies:
        singleBody.RK1()

    #RK2
    for singleBody in allBodies:
        singleBody.RK2()

    #RK3       
    for singleBody in allBodies:
        singleBody.RK3()

    #RK4       
    for singleBody in allBodies:
        singleBody.RK4()

    #RKFinal
    bodyIndex = 0        
    for singleBody in allBodies:
        singleBody.RKfinal()
        historyArray[bodyIndex].append([singleBody.x,singleBody.y,singleBody.v_x,singleBody.v_y,time])
        bodyIndex+=1

######Here the simulation is over and we can check the data######
####Check at what time each body completed one orbit (Crossed y axis twice)####
for i in range(1,9):
    crossings = 0
    planetTest = i
    planetName = allBodies[planetTest].name
    for i in range(0,len(historyArray[planetTest])-1):
        #We find the difference between the sun's y pos and our body position here
        currentY = historyArray[planetTest][i][1] - historyArray[0][i][1]
        nextY = historyArray[planetTest][i+1][1] - historyArray[0][i+1][1]
        currentV_Y = historyArray[planetTest][i][3]
     
        if((currentY<0) == ((nextY>0))):
            crossings+=1

        #This i is the last point before completing one orbit
        if(crossings == 2):
            #Interpolate that location to get a more accurate time based on it's velocity
            timeBefore = ((i*dt)/(60*60*24))
            interpTime = (abs(currentY)/abs(currentV_Y))/(60*60*24)
            print("Orbital period:",planetName,(timeBefore+interpTime)/365.26, "years")

            break   #Ignore rest of values, already have orbital period
    



#Actually plot the paths of all bodies from the historyArray of each
#Setup figure
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Relative X to sun/AU", fontsize=18)
ax1.set_ylabel("Relative Y to sun/AU", fontsize=18)
ax1.set_title("Free sun movement with v_y = -16", fontsize=18)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect(1)


#Plot each body's path
bodyIndex = 0
for singleBody in allBodies:
    name = singleBody.name
    colour = singleBody.colour
    xArray = []
    yArray = []
    
    for i in range(0,len(historyArray[1])):
        xArray.append(historyArray[bodyIndex][i][0]/AU)
        yArray.append(historyArray[bodyIndex][i][1]/AU)
    ax1.plot(xArray, yArray, color=colour, label=name, marker='o', ms=4)
    bodyIndex+=1

#Show and save the final image
ax1.legend(prop={'size': 18})
plt.savefig("NexactPlot.png")
plt.show()
