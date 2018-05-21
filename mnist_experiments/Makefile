
default_inp=10
default_width=25
default_depth=4


possible_inps := 5 10 25 100 500 784
possible_widths := 10 15 25 50 100
possible_depths := 2 3 4 5 6 7

inp_exp := $(foreach inp,$(possible_inps), nets/OddOrEven--$(inp)_inp--$(default_width)_width--$(default_depth)_depth)
width_exp := $(foreach width,$(possible_widths), nets/OddOrEven--$(default_inp)_inp--$(width)_width--$(default_depth)_depth)
depth_exp := $(foreach depth,$(possible_depths), nets/OddOrEven--$(default_inp)_inp--$(default_width)_width--$(depth)_depth)

all_exps := $(inp_exp) $(width_exp) $(depth_exp)

all_rlv := $(addsuffix /prop.rlv, $(all_exps))
all_min := $(addsuffix /min.txt, $(all_exps))

nets/OddOrEven%/prop.rlv:
	./generate_mnist_experiment.sh  $@

%/min.txt:%/prop.rlv
	../tools/opt_runner.py $< Approx $@

all: $(all_min)
