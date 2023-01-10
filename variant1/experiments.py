import qaoa_from_bitflip_exp as exp 

# Calls the experiment
# Evaluates formulas with n=10 variables and clause to variable ratio between 0.2 and 8
# => evaluates all integer m between 0.2*10 and 8*10 with stepsize 1

#(n, p, ratio_min, ratio_max, stepsize, num_per_step, warm_started_layers, save_dir):
p=2
exp.run_and_save(10, p, 0.2, 8, 1, 400, 0, 'results_n10p2')
p=4
exp.run_and_save(10, p, 0.2, 8, 1, 400, 0, 'results_n10p4')
p=6
exp.run_and_save(10, p, 0.2, 8, 1, 400, 0, 'results_n10p6')
p=8
exp.run_and_save(10, p, 0.2, 8, 1, 400, 0, 'results_n10p8')
p=10
exp.run_and_save(10, p, 0.2, 8, 1, 400, 0, 'results_n10p10')
