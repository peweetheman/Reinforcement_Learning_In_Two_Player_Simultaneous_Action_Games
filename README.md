# Meta-NASH and BRAT agents
# Patrick Phillips

Run
`pip install -r requirements.txt`

The files exposed in the main folder include the code for the META-NASH agent and the BRAT agent, as well as the code to run new tournaments. The "run_tournament" file is currently set up to run the Iterated Matching Pennies (IMP) tournament for 25 randomly  initialized  trials  each  consisting  of  50  episodes, each episode consisting of 100 transitions. The IMP tournament was run with γ= 1.0 and using Adam optimizer withlr= 0.005. These were the same settings used to generate the results in this paper.

To  run  tournaments  for  the  predator-prey  environment, there is a conflicting dependency for the gym package. I believe the only change that has to be made is to replace gym version .17.0 with .10.0, and then the predator-prey environment from the folder ma-gym should be available. 

To   plot   the   results   from   the   tournaments,   look   in the folders "matching-pennies_tournament" and "predator-prey_tournament".  In  each  folder  there  is  data  saved  in  a folder with the environment name, a .ipynb with the code to load the data and plot, and pdf images of the results.

The file CNN_context_dqn_nash_eq contains a version of the META-NASH algorithm suitable for environments that have a pixel state space such as the atari environments. No tests of this environment were included in the report.The "common" folder  includes  some  functionality  that  is useful to various algorithms implemented such as the replay buffer code.
