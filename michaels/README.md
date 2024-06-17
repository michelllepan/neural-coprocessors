# coproc-poc
Proof of concept of a neural co-processor, based on a simulated brain

## Layout

* train\_cpn.ipynb: contains a hook to start training a CPN+EN on the simulated brain, with a default choice of lesion design and observation function
* The other jupyter notebooks are bit rotted: they are exploratory, and contain some prior implementations of what is below

* experiment: contains the simulated "brain", which is a pytorch implementation of a network from Michaels et al.
  * mRNN.py - contains the actual Michaels "modular" RNN, some hooks to add a lesion and observation function, and a dataset loader. The data comes from Michaels directly.
  * experiment.py - controls the flow of a training session. It allows you to define your own model relatively easily by subclassing 'CoProc'. That model can be hooked in as you see  in the jupyter notebook above.
  * lesion.py - some lesion designs. These are pluggable, so one can design the lesion in various ways. I'm not sure how much that has bit rotted at this point.
  * michaels\_load.py - helper to load the Michaels training data as well as network weights.
  * observer.py - observation functions; again these are pluggable and tunable
  * stim.py - the stimulation model. As before: this is also pluggable.
  * config.py - used by the experment to bind the lesion, observation function, etc. to specific instances

* cpn.py - the CoProc defining the coprocessor design you saw in the slides: a CPN/EN approach that uses error backprop
* cpn\_model.py - defines the networks we use for the CPN
* stim\_model.py - defines the networks we use for the EN
* cpn\_epoch\_\* - these have the forward, backward, and stats tracking passes that cpn.py uses for training the CPN and EN

* micheals: a small subset of the data we received from Michaels, referenced by e.g. michaels\_load.py above
