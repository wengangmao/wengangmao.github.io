# Research Logbook -- Xiao

+++
---
Platform for VoySmart team to communicate our research activities in weekly meeting. Every week, we try to discuss research, education, teaching related duties, as well as results, achievements and problems during our research activities.

---

```{admonition}  <span style = "color: blue; font-weight: 600; font-size: 25px">Urgent duties</span>
<span style = "color: blue; font-weight: 400; font-size: 20px">Develop algorithms to optimize fixed speed/RPM/Power for optimal navigation.</span>
```

***
## Week 42 -- (Meeting on 2021-10-18)
---
***

### Plan/action for Week 42



***
## Week 41 -- (Meeting on 2021-10-11)
---
***

### Plan/action for Week 41
    
1. <span style = "font-weight: 400; font-size: 20px; color: blue">OE paper revision <span style = "background: Done">[by end of this week]<br /> </span></span>
2. <span style = "font-weight: 400; font-size: 20px; color: blue">Prepare for the Ecosail presentation: summarize your achievement so far <span style = "background: Done">[by end of this week]</span> <br /></span>
3. <span style = "font-weight: 400; font-size: 20px; color: blue">Submit NMU and ISOPE (if willing to China) papers <br /></span>
4. <span style = "font-weight: 400; font-size: 20px; color: blue">Assist Zhang Chi's paper revision <br /></span>

---
    




***
## Week 40 -- (Meeting on 2021-10-04)
---
***

### Plan/action for Week 40
    
1. <span style = "font-weight: 400; font-size: 20px; color: blue"><strike>OE paper revision <span style = "background: yellow">[by end of this week]</strike><br /> </span></span>
2. <span style = "font-weight: 400; font-size: 20px; color: blue">Prepare for the Ecosail presentation: summarize your achievement so far <br /></span>
3. <span style = "font-weight: 400; font-size: 20px; color: blue"><strike>Participant the Vattenfall project meeting</strike> <br /></span>

---
    


***
## Week 39 -- (Meeting on 2021-09-27)
---
***

### Plan/action for Week 39
    
1. OE paper revision <br /> 
2. <strike>Prepare meeting with Lean Marine <br /></strike>


---
    
    
### Meeting on 2021-09-30
Today, we had a meeting with LeanMarine to understand their FuelOpt and FleetAnalysis system. Their inputs will be valuable for our development of optimal navigation strategies. Good discussions with Linus from LeanMarine regarding the inputs of our future optimization development.
- SOG setting is not the optimal solution for energy efficiency shipping navigation
- For actual navigation, first legs: optimization of fixed RPM/power; last leg: speed fixed to reach ETA
- for sailing in harsh environments, by setting several control variables and constrain for max fuel/RPM 

```{figure} ./images/leanmarine210930/leanmarine_background.png 
---
height: 500px
name: lean_background
alt: leanmarine system
---
General background and possible setting of the leanmarine FuelOpt and FleetAnalysis system.
```



```{figure} ./images/leanmarine210930/leanmarine_fleetanalysis.png 
---
height: 600px
name: lean_background
alt: leanmarine system
---
Fleet analysis layout for the whole fleet of one shipping company called AUTO
```


```{figure} ./images/leanmarine210930/leanmarine_auto.png 
---
height: 500px
name: lean_background
alt: leanmarine system
---
An example of the performance monitoring of the fleetAnalysis of the AutoSKy ship
```


```{figure} ./images/leanmarine210930/leanmarine_data.png 
---
height: 400px
name: lean_background
alt: leanmarine system
---
Functionalities and GUI lookout of the FleetAnalysis system.
```



### General update of research activities

* Constant RPM and power for TTM vessels done with accurate ETA (error less than 1 hour)
* <span style = "color: blue; font-weight: 500">Completed tasks</span>
    > 1, Speed to power ML models <br />
    > 2, RPM to speed models <br />
    > 3, Power to speed models <br />
