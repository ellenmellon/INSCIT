This file contains instructions for creating human evaluationa input files, setting up AMT tasks and finally calculating human eval scores.

### Evaluation Input
By default, each evaluation task compares outputs from humans as well as two models. We can see preview the human evaluation interface through [this link](http://qa.cs.washington.edu:7782/task/validation/preview#). 

Running `python create_input.py` reads sample prediction files under `./sample` and creates the human evaluation input file at `./amt-human-eval/data/sources_eval.jsonl` Each prediction file has the same format as the automatic evaluation input file (those being generated in `../../results` if you ran our baseline models).

Since we currently decide to hide the test set, we do not release the human evaluation input and output files used in our paper for now. But we will give updates on this soon!


### AMT Setup
Our amt setup code is at `./amt-human-eval` and is based on [github.com/julianmichael/spacro](https://github.com/julianmichael/spacro). Please check it out for details of environment set up. Specifically, you need to install Mill and change lines 12-16 accordingly in `./scripts/initSample.scala`. You may also want to checkout [this repo](https://github.com/uwnlp/qamr/tree/main/code) to set up MTurk and make sure to have three files inside `sample/resources`: `logback.xml`, `<domain>-keystore-password` and `<domain>.p12`.

After the setup, you can simply do 
```
cd amt-human-eval
bash scripts/run_sample.sh
```

You will be able to see the preview at `http://<domain>:<httpPort>/task/validation/preview`. To launch the job in the sandbox, type `exp.start()` in your interactive shell. You will be able to see your jobs in `https://workersandbox.mturk.com`. Annotations that are submitted are saved in `data/example/main/live`. To change it from sandbox to production mode, you can set `val isProduction = true` in `./scripts/initSample.scala`.


After the annotion is all done, you can do `reset` and `:q` in the console.


#### Worker Qualification
We suggest that you launch a small batch to select highly qualified evaluators before running the actual evaluation. To do this, you can manually review their annotations and see check they make sense or not. Once you find a pool of qualified workers, you can type `grantQualification(workerId)` in the console for each of them. Then you can uncomment line 135 in `./sample/src-jvm/SampleExperiment.scala` to make sure all tasks are open to qualified workers only.



#### Setup Your Own Evaluation Task
If you are more comfortable with other crowd sourcing platforms or codebase, we still strongly encourage that you use similar interface and/or task instruction, which you can refer to files under `./js/` or [this link](http://qa.cs.washington.edu:7782/task/validation/preview#). 


### Evaluation Results
You can simply run `calc_human_eval_results.py` to get all human evaluation scores printed out, in the same format as in our paper. However, note that this assumes that you use our provided AMT tool. If you use some other tool to collect human evaluation, you may want to read the script and adapt the scoring functions to your own.
