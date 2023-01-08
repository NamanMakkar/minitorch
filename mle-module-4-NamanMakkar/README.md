# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py project/run_fast_tensor.py project/parallel_check.py tests/test_tensor_general.py


* Sentiment Classification

Epoch 1, loss 31.48819522134883, train accuracy: 48.44%

Validation accuracy: 51.00%

Best Valid accuracy: 51.00%

Epoch 2, loss 31.346360264618287, train accuracy: 48.67%

Validation accuracy: 49.00%

Best Valid accuracy: 51.00%

Epoch 3, loss 31.149382493130183, train accuracy: 52.22%

Validation accuracy: 50.00%

Best Valid accuracy: 51.00%

Epoch 4, loss 30.87900409740895, train accuracy: 53.56%

Validation accuracy: 56.00%

Best Valid accuracy: 56.00%

Epoch 5, loss 30.63505006542603, train accuracy: 58.89%

Validation accuracy: 55.00%

Best Valid accuracy: 56.00%

Epoch 6, loss 30.622873324346646, train accuracy: 56.00%

Validation accuracy: 56.00%

Best Valid accuracy: 56.00%

Epoch 7, loss 30.31276787008611, train accuracy: 59.56%

Validation accuracy: 54.00%

Best Valid accuracy: 56.00%

Epoch 8, loss 30.107232600495813, train accuracy: 62.00%

Validation accuracy: 60.00%

Best Valid accuracy: 60.00%

Epoch 9, loss 29.984232461015274, train accuracy: 62.89%

Validation accuracy: 56.00%

Best Valid accuracy: 60.00%

Epoch 10, loss 29.529745398931155, train accuracy: 66.89%

Validation accuracy: 54.00%

Best Valid accuracy: 60.00%

Epoch 11, loss 29.360202803360778, train accuracy: 64.89%

Validation accuracy: 59.00%

Best Valid accuracy: 60.00%

Epoch 12, loss 28.843203296460583, train accuracy: 66.22%

Validation accuracy: 61.00%

Best Valid accuracy: 61.00%

Epoch 13, loss 28.489627531507853, train accuracy: 63.33%

Validation accuracy: 60.00%

Best Valid accuracy: 61.00%

Epoch 14, loss 28.175764118542702, train accuracy: 70.22%

Validation accuracy: 61.00%

Best Valid accuracy: 61.00%

Epoch 15, loss 27.05172165610812, train accuracy: 72.00%

Validation accuracy: 66.00%

Best Valid accuracy: 66.00%

Epoch 16, loss 26.945638091379514, train accuracy: 70.00%

Validation accuracy: 57.00%

Best Valid accuracy: 66.00%

Epoch 17, loss 25.9323894041357, train accuracy: 74.44%

Validation accuracy: 62.00%

Best Valid accuracy: 66.00%

Epoch 18, loss 25.56097699289405, train accuracy: 74.44%

Validation accuracy: 59.00%

Best Valid accuracy: 66.00%

Epoch 19, loss 24.96178087179374, train accuracy: 74.22%

Validation accuracy: 66.00%

Best Valid accuracy: 66.00%

Epoch 20, loss 24.27664298134015, train accuracy: 76.00%

Validation accuracy: 66.00%

Best Valid accuracy: 66.00%

Epoch 21, loss 23.665217461520328, train accuracy: 77.11%

Validation accuracy: 65.00%

Best Valid accuracy: 66.00%

Epoch 22, loss 22.99342493029546, train accuracy: 74.89%

Validation accuracy: 63.00%

Best Valid accuracy: 66.00%

Epoch 23, loss 22.440193508731316, train accuracy: 77.11%

Validation accuracy: 72.00%

Best Valid accuracy: 72.00%

Epoch 24, loss 21.49010638902204, train accuracy: 78.22%

Validation accuracy: 71.00%

Best Valid accuracy: 72.00%

Epoch 25, loss 20.79757009181238, train accuracy: 79.11%

Validation accuracy: 65.00%

Best Valid accuracy: 72.00%

Epoch 26, loss 19.780018528219056, train accuracy: 82.22%

Validation accuracy: 64.00%

Best Valid accuracy: 72.00%

Epoch 27, loss 20.002753727285704, train accuracy: 78.89%

Validation accuracy: 68.00%

Best Valid accuracy: 72.00%

Epoch 28, loss 20.170070048734523, train accuracy: 79.78%

Validation accuracy: 64.00%

Best Valid accuracy: 72.00%

Epoch 29, loss 19.56162742548441, train accuracy: 79.11%

Validation accuracy: 66.00%

Best Valid accuracy: 72.00%

Epoch 30, loss 18.234217643362257, train accuracy: 80.89%

Validation accuracy: 62.00%

Best Valid accuracy: 72.00%

Epoch 31, loss 17.99478916718799, train accuracy: 82.00%

Validation accuracy: 72.00%

Best Valid accuracy: 72.00%

Epoch 32, loss 17.888353663570005, train accuracy: 81.56%

Validation accuracy: 69.00%

Best Valid accuracy: 72.00%

Epoch 33, loss 17.558572149952465, train accuracy: 81.33%

Validation accuracy: 63.00%

Best Valid accuracy: 72.00%

Epoch 34, loss 16.276966891673172, train accuracy: 85.11%

Validation accuracy: 66.00%

Best Valid accuracy: 72.00%

Epoch 35, loss 15.913360421952557, train accuracy: 84.67%

Validation accuracy: 70.00%

Best Valid accuracy: 72.00%

Epoch 36, loss 16.98457794503524, train accuracy: 82.00%

Validation accuracy: 66.00%

Best Valid accuracy: 72.00%

Epoch 37, loss 16.10501814651596, train accuracy: 83.56%

Validation accuracy: 64.00%

Best Valid accuracy: 72.00%

Epoch 38, loss 15.39590095251043, train accuracy: 84.67%

Validation accuracy: 66.00%

Best Valid accuracy: 72.00%

Epoch 39, loss 14.977070591178956, train accuracy: 85.11%

Validation accuracy: 67.00%

Best Valid accuracy: 72.00%

Epoch 40, loss 15.03528108169957, train accuracy: 85.33%

Validation accuracy: 67.00%

Best Valid accuracy: 72.00%

Epoch 41, loss 14.226586675964295, train accuracy: 84.67%

Validation accuracy: 66.00%

Best Valid accuracy: 72.00%

Epoch 42, loss 13.702882418439584, train accuracy: 88.00%

Validation accuracy: 64.00%

Best Valid accuracy: 72.00%

Epoch 43, loss 14.35927254383395, train accuracy: 85.56%

Validation accuracy: 76.00%

Best Valid accuracy: 76.00%



* MNIST Fast Conv Training - 

Epoch 1 loss 2.3242254323501887 valid acc 2/16
 
Epoch 1 loss 11.505670454827928 valid acc 2/16
 
Epoch 1 loss 11.494318856054049 valid acc 2/16
 
Epoch 1 loss 11.428783786002187 valid acc 4/16
 
Epoch 1 loss 11.457290744569477 valid acc 3/16
 
Epoch 1 loss 11.402987021365213 valid acc 2/16
 
Epoch 1 loss 11.179034222001299 valid acc 6/16
 
Epoch 1 loss 11.200467822345281 valid acc 6/16
 
Epoch 1 loss 10.988584228659 valid acc 8/16
 
Epoch 1 loss 10.341168059048698 valid acc 6/16
 
Epoch 1 loss 9.30775790258585 valid acc 7/16
 
Epoch 1 loss 9.129053111935164 valid acc 9/16
 
Epoch 1 loss 9.157440797811383 valid acc 10/16
 
Epoch 1 loss 8.625122371709734 valid acc 8/16
 
Epoch 1 loss 7.558915717210512 valid acc 8/16
 
Epoch 1 loss 8.59542971083648 valid acc 10/16
 
Epoch 1 loss 8.207634047927193 valid acc 10/16
 
Epoch 1 loss 6.8098019061853545 valid acc 8/16
 
Epoch 1 loss 6.668250765547839 valid acc 9/16
 
Epoch 1 loss 6.622967885538773 valid acc 11/16
 
Epoch 1 loss 7.026445439778548 valid acc 10/16
 
Epoch 1 loss 5.08496730783149 valid acc 13/16
 
Epoch 1 loss 2.7712972369818356 valid acc 10/16
 
Epoch 1 loss 5.534489804579773 valid acc 14/16
 
Epoch 1 loss 4.785098630584839 valid acc 10/16
 
Epoch 1 loss 5.063317273071798 valid acc 12/16
 
Epoch 1 loss 6.751963013696392 valid acc 10/16
 
Epoch 1 loss 3.361887199615131 valid acc 9/16
 
Epoch 1 loss 5.489411858316652 valid acc 13/16
 
Epoch 1 loss 2.9626889911833025 valid acc 12/16
 
Epoch 1 loss 7.245509195042667 valid acc 10/16
 
Epoch 1 loss 4.0481784698246965 valid acc 14/16
 
Epoch 1 loss 2.75892660404524 valid acc 13/16
 
Epoch 1 loss 4.583585831115808 valid acc 14/16
 
Epoch 1 loss 5.609250343372226 valid acc 13/16
 
Epoch 1 loss 4.122391418476426 valid acc 13/16
 
Epoch 1 loss 3.751755601476344 valid acc 12/16
 
Epoch 1 loss 3.9100034421702063 valid acc 15/16
 
Epoch 1 loss 3.890323043829145 valid acc 14/16
 
Epoch 1 loss 3.1509284613701274 valid acc 10/16
 
Epoch 1 loss 3.8173213743146324 valid acc 12/16
 
Epoch 1 loss 3.0607436736399105 valid acc 12/16
 
Epoch 1 loss 3.756066340477278 valid acc 11/16
 
Epoch 1 loss 2.2258814286142257 valid acc 13/16
 
Epoch 1 loss 3.8771525369689277 valid acc 14/16
 
Epoch 1 loss 1.891649511418584 valid acc 15/16
 
Epoch 1 loss 4.124697310223096 valid acc 13/16
 
Epoch 1 loss 3.163363705733901 valid acc 12/16
 
Epoch 1 loss 2.4835346248024037 valid acc 14/16
 
Epoch 1 loss 2.2489907809945167 valid acc 15/16
 
Epoch 1 loss 2.5862003494476147 valid acc 11/16
 
Epoch 1 loss 2.6720073548274197 valid acc 16/16
 
Epoch 1 loss 3.2430533809047466 valid acc 13/16
 
Epoch 1 loss 2.6127547333506116 valid acc 14/16
 
Epoch 1 loss 3.394712735227291 valid acc 14/16
 
Epoch 1 loss 2.9687749316683907 valid acc 15/16
 
Epoch 1 loss 3.275278543521577 valid acc 12/16
 
Epoch 1 loss 2.8039774388836802 valid acc 14/16
 
Epoch 1 loss 2.959629262082817 valid acc 14/16
 
Epoch 1 loss 3.738082332165006 valid acc 13/16
 
Epoch 1 loss 3.093617179710571 valid acc 14/16
 
Epoch 1 loss 3.3047026201388188 valid acc 14/16
 
Epoch 1 loss 2.631093771142524 valid acc 11/16
 
Epoch 2 loss 0.33501674565756095 valid acc 15/16
 
Epoch 2 loss 2.3075420645259936 valid acc 13/16
 
Epoch 2 loss 2.7086867282527765 valid acc 14/16
 
Epoch 2 loss 2.2261780735255767 valid acc 13/16
 
Epoch 2 loss 1.5263022921092095 valid acc 14/16
 
Epoch 2 loss 1.8754419316038555 valid acc 14/16
 
Epoch 2 loss 2.397519286027709 valid acc 13/16
 
Epoch 2 loss 3.2321281220049487 valid acc 13/16
 
Epoch 2 loss 2.751793335396034 valid acc 14/16
 
Epoch 2 loss 2.2411661779029735 valid acc 14/16
 
Epoch 2 loss 2.018386743765783 valid acc 15/16
 
Epoch 2 loss 3.849179415443899 valid acc 13/16
 
Epoch 2 loss 2.1675902207283304 valid acc 14/16
 
Epoch 2 loss 2.808368636210214 valid acc 14/16
 
Epoch 2 loss 2.4440574195443103 valid acc 14/16
 
Epoch 2 loss 1.6803050528194885 valid acc 14/16
 
Epoch 2 loss 3.6771283073407086 valid acc 13/16
 
Epoch 2 loss 2.5919589758093378 valid acc 14/16
 
Epoch 2 loss 2.8964284276368346 valid acc 13/16
 
Epoch 2 loss 1.2756266177686524 valid acc 14/16
 
Epoch 2 loss 2.53686739897872 valid acc 13/16
 
Epoch 2 loss 1.3054433687382125 valid acc 13/16
 
Epoch 2 loss 0.8250785585485328 valid acc 13/16
 
Epoch 2 loss 1.7176244533435714 valid acc 14/16
 
Epoch 2 loss 1.1877294684624276 valid acc 15/16
 
Epoch 2 loss 1.4723503380162062 valid acc 15/16
 
Epoch 2 loss 2.4673993762373145 valid acc 14/16
 
Epoch 2 loss 1.485886900769371 valid acc 14/16
 
Epoch 2 loss 2.227227795865333 valid acc 15/16
 
Epoch 2 loss 1.5402978288531097 valid acc 14/16
 
Epoch 2 loss 2.2337636499722877 valid acc 13/16
 
Epoch 2 loss 1.7454802619360796 valid acc 12/16
 
Epoch 2 loss 4.791668459104091 valid acc 14/16
 
Epoch 2 loss 2.444558122291801 valid acc 12/16
 
Epoch 2 loss 4.131904778819248 valid acc 14/16
 
Epoch 2 loss 1.792977334531535 valid acc 14/16
 
Epoch 2 loss 1.8453573444709002 valid acc 13/16
 
Epoch 2 loss 1.924111445915521 valid acc 13/16
 
Epoch 2 loss 1.9123431789749976 valid acc 14/16
 
Epoch 2 loss 1.939642436232622 valid acc 12/16
 
Epoch 2 loss 1.4514204699844804 valid acc 15/16
 
Epoch 2 loss 1.9547086644093523 valid acc 15/16
 
Epoch 2 loss 1.631956126481536 valid acc 15/16
 
Epoch 2 loss 1.4250103315665201 valid acc 15/16
 
Epoch 2 loss 2.4919086873225345 valid acc 15/16
 
Epoch 2 loss 0.568975205286608 valid acc 15/16
 
Epoch 2 loss 2.142371435015325 valid acc 16/16
 
Epoch 2 loss 2.0950760279954017 valid acc 16/16
 
Epoch 2 loss 0.8568530528558667 valid acc 14/16
 
Epoch 2 loss 0.7466605484973099 valid acc 15/16
 
Epoch 2 loss 1.3664489694265738 valid acc 13/16
 
Epoch 2 loss 1.266095298925394 valid acc 14/16
 
Epoch 2 loss 2.352134905208726 valid acc 14/16
 
Epoch 2 loss 0.7598977695748986 valid acc 15/16
 
Epoch 2 loss 2.713336846908097 valid acc 11/16
 
Epoch 2 loss 1.4002306780880647 valid acc 15/16
 
Epoch 2 loss 1.2742896307023008 valid acc 15/16
 
Epoch 2 loss 1.0368104258293966 valid acc 15/16
 
Epoch 2 loss 2.0201838578751743 valid acc 14/16
 
Epoch 2 loss 1.6797798681487428 valid acc 16/16
 
Epoch 2 loss 1.5457000752165768 valid acc 15/16
 
Epoch 2 loss 1.8226210798859606 valid acc 14/16
 
Epoch 2 loss 2.001483684303452 valid acc 15/16
 
Epoch 3 loss 0.1553999626781203 valid acc 15/16
 
Epoch 3 loss 1.574478396310266 valid acc 15/16
 
Epoch 3 loss 1.979580681593597 valid acc 14/16
 
Epoch 3 loss 1.6669262736184978 valid acc 13/16
 
Epoch 3 loss 0.7709675305208475 valid acc 15/16
 
Epoch 3 loss 1.5125002160340608 valid acc 15/16
 
Epoch 3 loss 1.8242487313072806 valid acc 13/16
 
Epoch 3 loss 1.8151681589485351 valid acc 15/16
 
Epoch 3 loss 1.6253401810817139 valid acc 13/16
 
Epoch 3 loss 1.2033718209181075 valid acc 14/16
 
Epoch 3 loss 0.8946011088372969 valid acc 14/16
 
Epoch 3 loss 2.0207691118477844 valid acc 14/16
 
Epoch 3 loss 2.1420963418190353 valid acc 15/16
 
Epoch 3 loss 2.175354379705998 valid acc 13/16
 
Epoch 3 loss 2.3584309097683516 valid acc 15/16
 
Epoch 3 loss 0.9918900408846985 valid acc 16/16
 
Epoch 3 loss 2.250245731282236 valid acc 15/16
 
Epoch 3 loss 2.1387091861503724 valid acc 13/16
 
Epoch 3 loss 1.9150212332199241 valid acc 15/16
 
Epoch 3 loss 1.0789657438801266 valid acc 13/16
 
Epoch 3 loss 1.6584303531056563 valid acc 15/16
 
Epoch 3 loss 0.8996863449977074 valid acc 15/16
 
Epoch 3 loss 0.32075687504767897 valid acc 14/16
 
Epoch 3 loss 1.6649033807681195 valid acc 14/16
 
Epoch 3 loss 0.829920482214738 valid acc 15/16
 
Epoch 3 loss 1.6943377644517525 valid acc 15/16
 
Epoch 3 loss 1.453688201491733 valid acc 14/16
 
Epoch 3 loss 1.074471304804607 valid acc 15/16
 
Epoch 3 loss 1.7949456726457793 valid acc 14/16
 
Epoch 3 loss 0.8106863051002634 valid acc 13/16
 
Epoch 3 loss 1.313764667078297 valid acc 14/16
 
Epoch 3 loss 1.0825974794197455 valid acc 14/16
 
Epoch 3 loss 0.44404377395901407 valid acc 14/16
 
Epoch 3 loss 1.917425670468306 valid acc 13/16
 
Epoch 3 loss 2.0366901827579094 valid acc 14/16
 
Epoch 3 loss 1.226112828476885 valid acc 15/16
 
Epoch 3 loss 0.9075469043205607 valid acc 15/16
 
Epoch 3 loss 1.4325181755194758 valid acc 14/16
 
Epoch 3 loss 1.316252719607451 valid acc 14/16
 
Epoch 3 loss 1.0647225063199797 valid acc 14/16
 
Epoch 3 loss 1.0667887509596712 valid acc 15/16
 
Epoch 3 loss 0.8241151604498484 valid acc 15/16
 
Epoch 3 loss 1.3552209196575542 valid acc 14/16
 
Epoch 3 loss 0.6974250649757223 valid acc 15/16
 
Epoch 3 loss 1.5903591465217422 valid acc 13/16
 
Epoch 3 loss 0.3539670815659877 valid acc 15/16
 
Epoch 3 loss 1.5142789541906816 valid acc 15/16
 
Epoch 3 loss 1.9181366030089415 valid acc 15/16
 
Epoch 3 loss 0.5484890356746976 valid acc 15/16
 
Epoch 3 loss 0.6132999417169269 valid acc 16/16
 
Epoch 3 loss 0.9000422671804169 valid acc 14/16
 
Epoch 3 loss 1.3591302832539185 valid acc 15/16
 
Epoch 3 loss 1.5394690464195477 valid acc 15/16
 
Epoch 3 loss 0.828023499343194 valid acc 16/16
 
Epoch 3 loss 1.6081208323613472 valid acc 15/16
 
Epoch 3 loss 0.7725073094707204 valid acc 15/16
 
Epoch 3 loss 1.8253351500826873 valid acc 15/16
 
Epoch 3 loss 0.6554672344229288 valid acc 14/16
 
Epoch 3 loss 1.2748443814499242 valid acc 14/16
 
Epoch 3 loss 1.6454988120519007 valid acc 15/16
 
Epoch 3 loss 0.687416989456195 valid acc 15/16
 
Epoch 3 loss 0.9068274034244469 valid acc 15/16
 
Epoch 3 loss 1.6517552120227306 valid acc 15/16
 
Epoch 4 loss 0.01945099903020432 valid acc 15/16
 
Epoch 4 loss 1.187231132588789 valid acc 15/16
 
Epoch 4 loss 1.0934319177789407 valid acc 15/16
 
Epoch 4 loss 1.3357608494601867 valid acc 15/16
 
Epoch 4 loss 0.6289106265797818 valid acc 15/16
 
Epoch 4 loss 0.6629947350886141 valid acc 16/16
 
Epoch 4 loss 1.4509559877618403 valid acc 14/16
 
Epoch 4 loss 1.394314313090967 valid acc 16/16
 
Epoch 4 loss 1.164089191249328 valid acc 15/16
 
Epoch 4 loss 0.7487824574813147 valid acc 15/16
 
Epoch 4 loss 0.5293203054211297 valid acc 16/16
 
Epoch 4 loss 1.8301149305868845 valid acc 14/16
 
Epoch 4 loss 1.443581202475145 valid acc 15/16
 
Epoch 4 loss 1.4929227629161645 valid acc 14/16
 
Epoch 4 loss 1.4993867615296113 valid acc 15/16
 
Epoch 4 loss 0.5711269683170687 valid acc 15/16
 
Epoch 4 loss 1.6861151910771854 valid acc 14/16
 
Epoch 4 loss 1.4927779836905044 valid acc 14/16
 
Epoch 4 loss 1.342692241111774 valid acc 15/16
 
Epoch 4 loss 0.7489866473832242 valid acc 15/16
 
Epoch 4 loss 1.0715439026472156 valid acc 15/16
 
Epoch 4 loss 0.8948218264744989 valid acc 15/16
 
Epoch 4 loss 0.21415617032719025 valid acc 15/16
 
Epoch 4 loss 1.2186342681989693 valid acc 15/16
 
Epoch 4 loss 0.27963319753835975 valid acc 15/16
 
Epoch 4 loss 1.1292187622038596 valid acc 15/16
 
Epoch 4 loss 1.1000963157590034 valid acc 15/16
 
Epoch 4 loss 0.5171925134387847 valid acc 15/16
 
Epoch 4 loss 1.0725939988168656 valid acc 15/16
 
Epoch 4 loss 0.5251111796558998 valid acc 15/16
 
Epoch 4 loss 1.3249303451085896 valid acc 15/16
 
Epoch 4 loss 0.6578516149543165 valid acc 15/16
 
Epoch 4 loss 0.3700612527499997 valid acc 15/16
 
Epoch 4 loss 0.9854903024854053 valid acc 16/16
 
Epoch 4 loss 2.0817228675651793 valid acc 13/16
 
Epoch 4 loss 1.1001251559255352 valid acc 15/16
 
Epoch 4 loss 0.3651868137166254 valid acc 15/16
 
Epoch 4 loss 0.7590294770178894 valid acc 15/16
 
Epoch 4 loss 0.41078136968776535 valid acc 15/16
 
Epoch 4 loss 1.0365417076095746 valid acc 15/16
 
Epoch 4 loss 0.7183060560186436 valid acc 15/16
 
Epoch 4 loss 0.9043996075923415 valid acc 15/16
 
Epoch 4 loss 0.8402828250895841 valid acc 15/16
 
Epoch 4 loss 0.5111639915441266 valid acc 15/16
 
Epoch 4 loss 1.1455944503405828 valid acc 15/16
 
Epoch 4 loss 0.4032575336154398 valid acc 15/16
 
Epoch 4 loss 1.0693081486429967 valid acc 15/16
 
Epoch 4 loss 1.0589992473215015 valid acc 14/16
 
Epoch 4 loss 0.5451695707876626 valid acc 15/16
 
Epoch 4 loss 0.36268898802958804 valid acc 16/16
 
Epoch 4 loss 1.3916027693646231 valid acc 15/16
 
Epoch 4 loss 1.4237192051820033 valid acc 14/16
 
Epoch 4 loss 1.4252220783681846 valid acc 15/16
 
Epoch 4 loss 0.7586639580387857 valid acc 14/16
 
Epoch 4 loss 1.0015307750673288 valid acc 14/16
 
Epoch 4 loss 1.1483281301304402 valid acc 15/16
 
Epoch 4 loss 1.56735103717714 valid acc 15/16
 
Epoch 4 loss 0.6854885308417197 valid acc 15/16
 
Epoch 4 loss 1.031917113574482 valid acc 14/16
 
Epoch 4 loss 1.0176391778382867 valid acc 14/16
 
Epoch 4 loss 0.7027156765275356 valid acc 15/16
 
Epoch 4 loss 0.5902274817914536 valid acc 15/16
 
Epoch 4 loss 1.3509089423977567 valid acc 15/16
 
Epoch 5 loss 0.17344185116700775 valid acc 15/16
 
Epoch 5 loss 0.7096067659620288 valid acc 15/16
 
Epoch 5 loss 1.1764787239519654 valid acc 15/16
 
Epoch 5 loss 0.9803045626194372 valid acc 13/16
 
Epoch 5 loss 0.37809427327437967 valid acc 15/16
 
Epoch 5 loss 0.6481660905122844 valid acc 14/16
 
Epoch 5 loss 0.9774618249734173 valid acc 16/16
 
Epoch 5 loss 1.3252322506324659 valid acc 15/16
 
Epoch 5 loss 0.7116996358289863 valid acc 15/16
 
Epoch 5 loss 0.8350100898484638 valid acc 15/16
 
Epoch 5 loss 0.2952966512000666 valid acc 15/16
 
Epoch 5 loss 1.9648421526733668 valid acc 15/16
 
Epoch 5 loss 1.2281727240641156 valid acc 14/16
 
Epoch 5 loss 1.3559859728527757 valid acc 16/16
 
Epoch 5 loss 0.9606033264063378 valid acc 14/16
 
Epoch 5 loss 0.9080711396070522 valid acc 15/16
 
Epoch 5 loss 0.9834658279666366 valid acc 15/16
 
Epoch 5 loss 1.435797745462561 valid acc 15/16
 
Epoch 5 loss 1.283833197117507 valid acc 15/16
 
Epoch 5 loss 0.3809289715742149 valid acc 15/16
 
Epoch 5 loss 0.6719928642903233 valid acc 15/16
 
Epoch 5 loss 1.0154369899686038 valid acc 14/16
 
Epoch 5 loss 0.4096075296661155 valid acc 15/16
 
Epoch 5 loss 1.5747342022093356 valid acc 14/16
 
Epoch 5 loss 0.5238212728955955 valid acc 15/16
 
Epoch 5 loss 1.209792981645851 valid acc 15/16
 
Epoch 5 loss 1.0138435286618117 valid acc 15/16
 
Epoch 5 loss 0.6225091987303875 valid acc 16/16
 
Epoch 5 loss 0.8479380195121176 valid acc 15/16
 
Epoch 5 loss 0.23547540476040868 valid acc 16/16
 
Epoch 5 loss 0.7336734432904073 valid acc 15/16
 
Epoch 5 loss 0.7277761825454 valid acc 14/16
 
Epoch 5 loss 0.3040437581027927 valid acc 15/16
 
Epoch 5 loss 1.4416203078596395 valid acc 15/16
 
Epoch 5 loss 1.7238074424540522 valid acc 15/16
 
Epoch 5 loss 0.7280682436300638 valid acc 15/16
 
Epoch 5 loss 0.8806659965198421 valid acc 15/16
 
Epoch 5 loss 0.6868665804648132 valid acc 15/16
 
Epoch 5 loss 0.49572806515346646 valid acc 15/16
 
Epoch 5 loss 0.25699771755858497 valid acc 15/16
 
Epoch 5 loss 0.4298571039283938 valid acc 15/16
 
Epoch 5 loss 0.6698818987798775 valid acc 14/16
 
Epoch 5 loss 0.6388876268891808 valid acc 15/16
 
Epoch 5 loss 0.32964912772475313 valid acc 15/16
 
Epoch 5 loss 0.5771697113604045 valid acc 16/16
 
Epoch 5 loss 0.15510648127954543 valid acc 16/16
 
Epoch 5 loss 0.7166260214351171 valid acc 15/16
 
Epoch 5 loss 0.7844880746736266 valid acc 16/16
 
Epoch 5 loss 0.26899206413109783 valid acc 15/16
 
Epoch 5 loss 1.0047317183516677 valid acc 15/16
 
Epoch 5 loss 0.228522999359951 valid acc 14/16
 
Epoch 5 loss 1.0887594581741076 valid acc 14/16
 
Epoch 5 loss 1.3666558734233334 valid acc 15/16
 
Epoch 5 loss 0.33172055581701276 valid acc 15/16
 
Epoch 5 loss 0.5917304502021086 valid acc 15/16
 
Epoch 5 loss 0.6966215780597224 valid acc 15/16
 
Epoch 5 loss 1.0873717275524106 valid acc 16/16
 
Epoch 5 loss 0.3482105743469383 valid acc 15/16
 
Epoch 5 loss 1.3017870054750948 valid acc 15/16
 
Epoch 5 loss 0.46531387658929096 valid acc 15/16
 
Epoch 5 loss 0.46419804902664386 valid acc 15/16
 
Epoch 5 loss 0.6406005201020799 valid acc 15/16
 
Epoch 5 loss 0.5916894592558901 valid acc 14/16
 
Epoch 6 loss 0.02077092867152043 valid acc 15/16
 
Epoch 6 loss 0.6818394723012975 valid acc 15/16
 
Epoch 6 loss 0.6951600904830968 valid acc 15/16
 
Epoch 6 loss 1.395755104129368 valid acc 15/16
 
Epoch 6 loss 0.7697945011375875 valid acc 14/16
 
Epoch 6 loss 0.5675490054842857 valid acc 15/16
 
Epoch 6 loss 1.9329516287249562 valid acc 13/16
 
Epoch 6 loss 0.9300043069424971 valid acc 13/16
 
Epoch 6 loss 0.7910282703628156 valid acc 16/16
 
Epoch 6 loss 0.42006297985658686 valid acc 15/16
 
Epoch 6 loss 0.1880178068331225 valid acc 15/16
 
Epoch 6 loss 1.7984443820836213 valid acc 14/16
 
Epoch 6 loss 0.9656632708748997 valid acc 15/16
 
Epoch 6 loss 1.1626954535815996 valid acc 12/16
 
Epoch 6 loss 0.9206691556029365 valid acc 15/16
 
Epoch 6 loss 0.6497693494815973 valid acc 15/16
 
Epoch 6 loss 1.9103765391260148 valid acc 14/16
 
Epoch 6 loss 1.8269436492328457 valid acc 15/16
 
Epoch 6 loss 0.7037464795609586 valid acc 14/16
 
Epoch 6 loss 0.8655432265464008 valid acc 15/16
 
Epoch 6 loss 1.0535556716608199 valid acc 15/16
 
Epoch 6 loss 0.40468069555696046 valid acc 14/16
 
Epoch 6 loss 0.1526151592266724 valid acc 15/16
 
Epoch 6 loss 0.329713426128982 valid acc 14/16
 
Epoch 6 loss 0.3102135519517232 valid acc 14/16
 
Epoch 6 loss 0.764980979349438 valid acc 15/16
 
Epoch 6 loss 0.6179706809451565 valid acc 15/16
 
Epoch 6 loss 0.4881675339862257 valid acc 16/16
 
Epoch 6 loss 0.658170887088653 valid acc 15/16
 
Epoch 6 loss 0.3344974069306445 valid acc 15/16
 
Epoch 6 loss 0.7831526414904417 valid acc 15/16
 
Epoch 6 loss 0.43496813208507956 valid acc 15/16
 
Epoch 6 loss 0.16064739308472603 valid acc 15/16
 
Epoch 6 loss 0.6245381373998404 valid acc 15/16
 
Epoch 6 loss 1.867101782113623 valid acc 15/16
 
Epoch 6 loss 0.6429860284090737 valid acc 15/16
 
Epoch 6 loss 0.4333039562098604 valid acc 15/16
 
Epoch 6 loss 0.7875679649199109 valid acc 15/16
 
Epoch 6 loss 0.4068417639774767 valid acc 13/16
 
Epoch 6 loss 0.5205945949009793 valid acc 15/16
 
Epoch 6 loss 0.3560104643827535 valid acc 16/16
 
Epoch 6 loss 0.2839549092985865 valid acc 16/16
 
Epoch 6 loss 0.5689425881774046 valid acc 15/16
 
Epoch 6 loss 0.14894527127949903 valid acc 15/16
 
Epoch 6 loss 0.4360673880345661 valid acc 16/16
 
Epoch 6 loss 0.10802275214928218 valid acc 16/16
 
Epoch 6 loss 0.6397637420132639 valid acc 15/16
 
Epoch 6 loss 1.2416231579543744 valid acc 14/16
 
Epoch 6 loss 0.2618103885477982 valid acc 16/16
 
Epoch 6 loss 0.3800656554738639 valid acc 16/16
 
Epoch 6 loss 0.48474192541453975 valid acc 16/16
 
Epoch 6 loss 0.25620482551817236 valid acc 16/16
 
Epoch 6 loss 0.6878752162842328 valid acc 15/16
 
Epoch 6 loss 0.5841104326449196 valid acc 15/16
 
Epoch 6 loss 0.8746098224596257 valid acc 15/16
 
Epoch 6 loss 0.6407404552909385 valid acc 14/16
 
Epoch 6 loss 0.9550476572175475 valid acc 13/16
 
Epoch 6 loss 0.5781183861976593 valid acc 14/16
 
Epoch 6 loss 0.9986011368752505 valid acc 14/16
 
Epoch 6 loss 0.7022105035782896 valid acc 14/16
 
Epoch 6 loss 0.3170519506280299 valid acc 15/16
 
Epoch 6 loss 0.5817485397276814 valid acc 14/16
 
Epoch 6 loss 0.6561097337759476 valid acc 14/16
 
Epoch 7 loss 0.017906941180181872 valid acc 15/16
 
Epoch 7 loss 0.4654850012850822 valid acc 15/16
 
Epoch 7 loss 1.2300716343987097 valid acc 15/16
 
Epoch 7 loss 0.6235404425650055 valid acc 15/16
 
Epoch 7 loss 0.06693945021665207 valid acc 15/16
 
Epoch 7 loss 0.4659418890698084 valid acc 15/16
 
Epoch 7 loss 0.5070500741645997 valid acc 15/16
 
Epoch 7 loss 0.9995451676346789 valid acc 14/16
 
Epoch 7 loss 0.5651385440779578 valid acc 15/16
 
Epoch 7 loss 0.9055768426963797 valid acc 15/16
 
Epoch 7 loss 0.20274881851223422 valid acc 16/16
 
Epoch 7 loss 1.0734789950852548 valid acc 15/16
 
Epoch 7 loss 0.8044384119614122 valid acc 15/16
 
Epoch 7 loss 0.698484266567871 valid acc 14/16
 
Epoch 7 loss 0.6424344457389921 valid acc 15/16
 
Epoch 7 loss 0.6304904185469298 valid acc 15/16
 
Epoch 7 loss 0.5340685239424112 valid acc 15/16
 
Epoch 7 loss 0.6248523321133423 valid acc 15/16
 
Epoch 7 loss 0.8215451469193208 valid acc 15/16
 
Epoch 7 loss 0.3967010226965119 valid acc 15/16
 
Epoch 7 loss 0.38817294235855077 valid acc 15/16
 
Epoch 7 loss 0.5342295891360058 valid acc 15/16
 
Epoch 7 loss 0.1512633786337871 valid acc 14/16
 
Epoch 7 loss 1.0576769107948258 valid acc 13/16
 
Epoch 7 loss 0.6147371101499399 valid acc 13/16
 
Epoch 7 loss 0.6571830885227405 valid acc 15/16
 
Epoch 7 loss 0.5378463650691115 valid acc 15/16
 
Epoch 7 loss 0.11821310472877417 valid acc 15/16
 
Epoch 7 loss 0.3952725726910187 valid acc 15/16
 
Epoch 7 loss 0.27471690835937224 valid acc 15/16
 
Epoch 7 loss 1.174803084019807 valid acc 14/16
 
Epoch 7 loss 0.2927603755552405 valid acc 13/16
 
Epoch 7 loss 0.45333041100417476 valid acc 15/16
 
Epoch 7 loss 1.095145237576465 valid acc 14/16
 
Epoch 7 loss 1.857856860854305 valid acc 13/16
 
Epoch 7 loss 0.6161731488686244 valid acc 14/16
 
Epoch 7 loss 1.184566910780125 valid acc 15/16
 
Epoch 7 loss 1.1149721732061182 valid acc 15/16
 
Epoch 7 loss 0.6880757963410753 valid acc 16/16
 
Epoch 7 loss 0.5123590776622486 valid acc 15/16
 
Epoch 7 loss 0.32500856516670423 valid acc 14/16
 
Epoch 7 loss 0.25106266172359876 valid acc 15/16
 
Epoch 7 loss 0.7233444781137189 valid acc 14/16
 
Epoch 7 loss 0.4617064797798555 valid acc 14/16
 
Epoch 7 loss 0.42165712227935254 valid acc 13/16
 
Epoch 7 loss 0.23416728312783744 valid acc 14/16
 
Epoch 7 loss 0.7888713925390733 valid acc 15/16
 
Epoch 7 loss 0.5206893503629889 valid acc 15/16
 
Epoch 7 loss 0.44182803308326 valid acc 15/16
 
Epoch 7 loss 0.18113174978041047 valid acc 16/16
 
Epoch 7 loss 0.29558225290741036 valid acc 15/16
 
Epoch 7 loss 0.9234537725858891 valid acc 15/16
 
Epoch 7 loss 1.0058283247237074 valid acc 15/16
 
Epoch 7 loss 0.28116818354723067 valid acc 15/16
 
Epoch 7 loss 0.6362314538649547 valid acc 14/16
 
Epoch 7 loss 0.7831947529244532 valid acc 15/16
 
Epoch 7 loss 1.1101548028826127 valid acc 15/16
 
Epoch 7 loss 0.2645986209566081 valid acc 15/16
 
Epoch 7 loss 0.8445761017626827 valid acc 16/16
 
Epoch 7 loss 0.42479197778632016 valid acc 16/16
 
Epoch 7 loss 0.4490420300260271 valid acc 15/16
 
Epoch 7 loss 0.35872534108865245 valid acc 15/16
 
Epoch 7 loss 0.9769164440485352 valid acc 16/16
 
Epoch 8 loss 0.0063016568636868305 valid acc 14/16
 
Epoch 8 loss 0.659476577710161 valid acc 15/16
 
Epoch 8 loss 0.6925625535504087 valid acc 16/16
 
Epoch 8 loss 0.6847885963969134 valid acc 15/16
 
Epoch 8 loss 0.5876800452416517 valid acc 15/16
 
Epoch 8 loss 0.8663776637101707 valid acc 15/16
 
Epoch 8 loss 0.7555467860286283 valid acc 15/16
 
Epoch 8 loss 1.37627490068345 valid acc 15/16
 
Epoch 8 loss 0.3959564784564957 valid acc 15/16
 
Epoch 8 loss 0.30783749719176456 valid acc 16/16
 
Epoch 8 loss 0.19866135995506493 valid acc 15/16
 
Epoch 8 loss 1.1146522307812574 valid acc 15/16
 
Epoch 8 loss 0.26109066312845136 valid acc 15/16
 
Epoch 8 loss 0.6686245601511437 valid acc 14/16
 
Epoch 8 loss 0.20085264172614176 valid acc 15/16
 
Epoch 8 loss 0.3609932703316881 valid acc 15/16
 
Epoch 8 loss 1.4781608430524733 valid acc 16/16
 
Epoch 8 loss 0.6578795886431452 valid acc 16/16
 
Epoch 8 loss 1.2843308432391087 valid acc 15/16
 
Epoch 8 loss 0.10793484253748216 valid acc 16/16
 
Epoch 8 loss 0.7867234046567518 valid acc 16/16
 
Epoch 8 loss 0.5648067010566409 valid acc 15/16
 
Epoch 8 loss 0.1568434361811421 valid acc 15/16
 
Epoch 8 loss 0.9445633962006625 valid acc 14/16
 
Epoch 8 loss 0.39285114597918347 valid acc 13/16
 
Epoch 8 loss 0.9497979608547542 valid acc 15/16
 
Epoch 8 loss 1.3735299377903674 valid acc 14/16
 
Epoch 8 loss 0.37684738273062357 valid acc 16/16
 
Epoch 8 loss 0.6828764764867936 valid acc 16/16
 
Epoch 8 loss 0.6792198858986552 valid acc 15/16
 
Epoch 8 loss 0.8060151205547801 valid acc 15/16
 
Epoch 8 loss 1.4499357299611255 valid acc 15/16
 
Epoch 8 loss 0.24846537790078338 valid acc 14/16
 
Epoch 8 loss 0.6169713062312654 valid acc 15/16
 
Epoch 8 loss 1.0421060832424427 valid acc 16/16
 
Epoch 8 loss 0.297611991516378 valid acc 16/16
 
Epoch 8 loss 0.22607365899441872 valid acc 16/16
 
Epoch 8 loss 0.7360454737422311 valid acc 15/16
 
Epoch 8 loss 0.7739078219623162 valid acc 16/16
 
Epoch 8 loss 0.12498957066986296 valid acc 16/16
 
Epoch 8 loss 0.19518002825857905 valid acc 16/16
 
Epoch 8 loss 0.39050488435052855 valid acc 16/16
 
Epoch 8 loss 0.5768682337493806 valid acc 15/16
 
Epoch 8 loss 0.33330621618300527 valid acc 15/16
 
Epoch 8 loss 0.6079119725711104 valid acc 15/16
 
Epoch 8 loss 0.10428343480013182 valid acc 14/16
 
Epoch 8 loss 0.5653555476631282 valid acc 16/16
 
Epoch 8 loss 0.7769577468337683 valid acc 16/16
 
Epoch 8 loss 0.3765327914564425 valid acc 15/16
 
Epoch 8 loss 0.16749259157193214 valid acc 16/16
 
Epoch 8 loss 0.3348521652584001 valid acc 16/16
 
Epoch 8 loss 0.6580219453536358 valid acc 16/16
 
Epoch 8 loss 0.39732916769098076 valid acc 15/16
 
Epoch 8 loss 0.2497941824870428 valid acc 15/16
 
Epoch 8 loss 0.2956748876274015 valid acc 14/16
 
Epoch 8 loss 1.0859898820033622 valid acc 15/16
 
Epoch 8 loss 0.7080043695918296 valid acc 14/16
 
Epoch 8 loss 0.2995112048242656 valid acc 14/16
 
Epoch 8 loss 0.6605443462609608 valid acc 14/16
 
Epoch 8 loss 0.6504965075259731 valid acc 16/16
 
Epoch 8 loss 0.2338409881508329 valid acc 16/16
 
Epoch 8 loss 0.44631263748495825 valid acc 15/16
 
Epoch 8 loss 0.535902878224263 valid acc 16/16
 
Epoch 9 loss 0.02010351180954464 valid acc 15/16
 
Epoch 9 loss 0.3761347597463058 valid acc 15/16
 
Epoch 9 loss 0.9059150941949492 valid acc 16/16
 
Epoch 9 loss 0.5069138778706938 valid acc 15/16
 
Epoch 9 loss 0.38427307884634027 valid acc 14/16
 
Epoch 9 loss 0.10933986146285285 valid acc 15/16
 
Epoch 9 loss 0.808293970687229 valid acc 16/16
 
Epoch 9 loss 0.6566766967649533 valid acc 15/16
 
Epoch 9 loss 0.6055285072165884 valid acc 15/16
 
Epoch 9 loss 0.2179050849431569 valid acc 16/16
 
Epoch 9 loss 0.3008215821214747 valid acc 16/16
 
Epoch 9 loss 0.48752347754712644 valid acc 16/16
 
Epoch 9 loss 0.6022966984116991 valid acc 14/16
 
Epoch 9 loss 0.6513008160959942 valid acc 15/16
 
Epoch 9 loss 0.8822939433188726 valid acc 15/16
 
Epoch 9 loss 0.4931837237877085 valid acc 15/16
 
Epoch 9 loss 1.0221531441556142 valid acc 15/16
 
Epoch 9 loss 1.0157007279782566 valid acc 14/16
 
Epoch 9 loss 0.4872437866282594 valid acc 15/16
 
Epoch 9 loss 0.4934761338090429 valid acc 14/16
 
Epoch 9 loss 0.5160174037715676 valid acc 15/16
 
Epoch 9 loss 0.4079028048125206 valid acc 15/16
 
Epoch 9 loss 0.3164315020797441 valid acc 15/16
 
Epoch 9 loss 0.45679413265085045 valid acc 14/16
 
Epoch 9 loss 0.47499156715954993 valid acc 13/16
 
Epoch 9 loss 0.9622101818056648 valid acc 16/16
 
Epoch 9 loss 0.7199506560532957 valid acc 15/16
 
Epoch 9 loss 0.1331276316072993 valid acc 15/16
 
Epoch 9 loss 0.4420129205425482 valid acc 15/16
 
Epoch 9 loss 0.6193455279653499 valid acc 15/16
 
Epoch 9 loss 0.07995902444160007 valid acc 16/16
 
Epoch 9 loss 0.14112759342892256 valid acc 15/16
 
Epoch 9 loss 0.6676148873996828 valid acc 14/16
 
Epoch 9 loss 1.4217811739348245 valid acc 15/16
 
Epoch 9 loss 0.9030803712300879 valid acc 15/16
 
Epoch 9 loss 0.28644850364163743 valid acc 15/16
 
Epoch 9 loss 0.16959351098090059 valid acc 15/16
 
Epoch 9 loss 0.8006497110146884 valid acc 15/16
 
Epoch 9 loss 0.13658707523620994 valid acc 15/16
 
Epoch 9 loss 0.47294902083970286 valid acc 14/16
 
Epoch 9 loss 0.17302847895076112 valid acc 15/16
 
Epoch 9 loss 0.6172362399263507 valid acc 15/16
 
Epoch 9 loss 0.3278189611546738 valid acc 15/16
 
Epoch 9 loss 0.19357674045565917 valid acc 14/16
 
Epoch 9 loss 0.8592473184979799 valid acc 14/16
 
Epoch 9 loss 0.5184451955611273 valid acc 14/16
 
Epoch 9 loss 0.6708444357095902 valid acc 14/16
 
Epoch 9 loss 0.5255517912268635 valid acc 16/16
 
Epoch 9 loss 0.33310256966849044 valid acc 15/16
 
Epoch 9 loss 0.16572475952819993 valid acc 15/16
 
Epoch 9 loss 0.7339983962347184 valid acc 16/16
 
Epoch 9 loss 1.2227627917251331 valid acc 15/16
 
Epoch 9 loss 0.7456356606814639 valid acc 14/16
 
Epoch 9 loss 0.3208340064681086 valid acc 15/16
 
Epoch 9 loss 0.47720002043588866 valid acc 14/16
 
Epoch 9 loss 0.14549011776768772 valid acc 15/16
 
Epoch 9 loss 0.8418583156847589 valid acc 15/16
 
Epoch 9 loss 0.1632517057608777 valid acc 15/16
 
Epoch 9 loss 0.5571020944005968 valid acc 14/16
 
Epoch 9 loss 0.5246266421707081 valid acc 14/16
 
Epoch 9 loss 0.13237043438929177 valid acc 15/16
 
Epoch 9 loss 0.27531943326242536 valid acc 15/16
 
Epoch 9 loss 0.7515224825068967 valid acc 15/16
 
Epoch 10 loss 0.062204594174764545 valid acc 15/16
 
Epoch 10 loss 0.25678349751876073 valid acc 14/16
 
Epoch 10 loss 0.21504945631960348 valid acc 15/16
 
Epoch 10 loss 0.6377427382937695 valid acc 15/16
 
Epoch 10 loss 0.07146246102096197 valid acc 15/16
 
Epoch 10 loss 0.08578281417237596 valid acc 14/16
 
Epoch 10 loss 0.08115641697276382 valid acc 15/16
 
Epoch 10 loss 0.3609615998576179 valid acc 16/16
 
Epoch 10 loss 0.3571845792124594 valid acc 15/16
 
Epoch 10 loss 0.6058949085600475 valid acc 16/16
 
Epoch 10 loss 0.21894711079123122 valid acc 16/16
 
Epoch 10 loss 0.6225027961432372 valid acc 15/16
 
Epoch 10 loss 0.3047806439362057 valid acc 15/16
 
Epoch 10 loss 1.0733199715447244 valid acc 15/16
 
Epoch 10 loss 1.1877910424610931 valid acc 14/16
 
Epoch 10 loss 0.36004728560179206 valid acc 14/16
 
Epoch 10 loss 1.2842302994591148 valid acc 14/16
 
Epoch 10 loss 0.8632695873481284 valid acc 14/16
 
Epoch 10 loss 0.5644035898064754 valid acc 15/16
 
Epoch 10 loss 0.21222920369358358 valid acc 14/16
 
Epoch 10 loss 0.4327910408362362 valid acc 15/16
 
Epoch 10 loss 0.2317374832211388 valid acc 15/16
 
Epoch 10 loss 0.07808603377211622 valid acc 14/16
 
Epoch 10 loss 0.5627277608722037 valid acc 15/16
 
Epoch 10 loss 0.17743414670990912 valid acc 14/16
 
Epoch 10 loss 0.6785794413172952 valid acc 14/16
 
Epoch 10 loss 0.7375554493504528 valid acc 15/16
 
Epoch 10 loss 0.19048509159554586 valid acc 14/16
 
Epoch 10 loss 0.47803381158391584 valid acc 16/16
 
Epoch 10 loss 0.26573364547814493 valid acc 14/16
 
Epoch 10 loss 0.5246283572605119 valid acc 15/16
 
Epoch 10 loss 0.6100152618320047 valid acc 14/16
 
Epoch 10 loss 0.4515901801526653 valid acc 14/16
 
Epoch 10 loss 0.20751185655090634 valid acc 15/16
 
Epoch 10 loss 0.9134093475178777 valid acc 15/16
 
Epoch 10 loss 0.21481754507611522 valid acc 15/16
 
Epoch 10 loss 0.5777276086105956 valid acc 14/16
 
Epoch 10 loss 0.6534336591446902 valid acc 14/16
 
Epoch 10 loss 0.35973164388596035 valid acc 14/16
 
Epoch 10 loss 0.23975993855924874 valid acc 15/16
 
Epoch 10 loss 0.3537561667215007 valid acc 15/16
 
Epoch 10 loss 0.24845858358280776 valid acc 15/16
 
Epoch 10 loss 0.46582996543051886 valid acc 15/16
 
Epoch 10 loss 0.2060238155142769 valid acc 15/16
 
Epoch 10 loss 0.3478780015249721 valid acc 16/16
 
Epoch 10 loss 0.027477582276534053 valid acc 16/16
 
Epoch 10 loss 0.7082728223942524 valid acc 15/16
 
Epoch 10 loss 0.8519615676414086 valid acc 15/16
 
Epoch 10 loss 0.13428408473125636 valid acc 16/16
 
Epoch 10 loss 0.16895337663058765 valid acc 16/16
 
Epoch 10 loss 0.5140679855908501 valid acc 15/16
 
Epoch 10 loss 0.18877215861251306 valid acc 15/16
 
Epoch 10 loss 0.37947809417018763 valid acc 15/16
 
Epoch 10 loss 0.269094727760258 valid acc 15/16
 
Epoch 10 loss 0.47366414743593976 valid acc 15/16
 
Epoch 10 loss 0.3910361138451406 valid acc 15/16
 
Epoch 10 loss 0.5896907645432736 valid acc 15/16
 
Epoch 10 loss 0.2695954370146732 valid acc 15/16
 
Epoch 10 loss 0.6218916094563289 valid acc 14/16
 
Epoch 10 loss 0.3498710583454212 valid acc 16/16
 
Epoch 10 loss 0.14057984801757936 valid acc 16/16
 
Epoch 10 loss 0.18005820608426681 valid acc 14/16
 
Epoch 10 loss 0.3643565427589637 valid acc 15/16
 
Epoch 11 loss 0.0004188630461107254 valid acc 15/16
 
Epoch 11 loss 0.3704601923581847 valid acc 16/16
 
Epoch 11 loss 0.6034671507965184 valid acc 14/16
 
Epoch 11 loss 0.5724950983687852 valid acc 14/16
 
Epoch 11 loss 0.05785077583962131 valid acc 15/16
 
Epoch 11 loss 0.17676174264091038 valid acc 15/16
 
Epoch 11 loss 0.954368726069678 valid acc 15/16
 
Epoch 11 loss 1.4155904724721713 valid acc 14/16
 
Epoch 11 loss 1.1053823838118282 valid acc 15/16
 
Epoch 11 loss 0.4353638572008257 valid acc 15/16
 
Epoch 11 loss 0.15641101313600922 valid acc 16/16
 
Epoch 11 loss 1.4015043084927916 valid acc 14/16
 
Epoch 11 loss 0.8846810604440012 valid acc 15/16
 
Epoch 11 loss 0.6398580582375255 valid acc 13/16
 
Epoch 11 loss 0.14722904774770104 valid acc 14/16
 
Epoch 11 loss 0.6452872269079057 valid acc 13/16
 
Epoch 11 loss 0.8742526302771804 valid acc 14/16
 
Epoch 11 loss 0.3367376834045996 valid acc 15/16
 
Epoch 11 loss 0.5181901947797412 valid acc 15/16
 
Epoch 11 loss 0.24990835772341272 valid acc 15/16
 
Epoch 11 loss 0.4116431375540847 valid acc 12/16
 
Epoch 11 loss 0.23104783498784479 valid acc 15/16
 
Epoch 11 loss 0.23389468853245704 valid acc 15/16
 
Epoch 11 loss 0.4257666239461837 valid acc 12/16
 
Epoch 11 loss 0.7418263445834845 valid acc 14/16
 
Epoch 11 loss 0.5253290197351723 valid acc 14/16
 
Epoch 11 loss 0.25532331243148104 valid acc 14/16
 
Epoch 11 loss 0.21870869685437866 valid acc 16/16
 
Epoch 11 loss 0.18927704627568165 valid acc 15/16
 
Epoch 11 loss 0.08137559950801224 valid acc 16/16
 
Epoch 11 loss 0.056819663987903155 valid acc 15/16
 
Epoch 11 loss 0.09125908678801524 valid acc 14/16
 
Epoch 11 loss 0.13660034735949214 valid acc 15/16
 
Epoch 11 loss 0.437710879187047 valid acc 15/16
 
Epoch 11 loss 1.0941650840592088 valid acc 14/16
 
Epoch 11 loss 0.19870345264317196 valid acc 15/16
 
Epoch 11 loss 0.09713606801622238 valid acc 15/16
 
Epoch 11 loss 0.7107076162799136 valid acc 16/16
 
Epoch 11 loss 0.06662826460569661 valid acc 14/16
 
Epoch 11 loss 0.9358513611171237 valid acc 14/16
 
Epoch 11 loss 0.4700594730332138 valid acc 15/16
 
Epoch 11 loss 0.49686546917887686 valid acc 15/16
 
Epoch 11 loss 0.5286585825224182 valid acc 15/16
 
Epoch 11 loss 0.3707840245487813 valid acc 13/16
 
Epoch 11 loss 0.9783370302663809 valid acc 16/16
 
Epoch 11 loss 0.27279630883568884 valid acc 14/16
 
Epoch 11 loss 0.32819212084786586 valid acc 15/16
 
Epoch 11 loss 0.5850615045019748 valid acc 15/16
 
Epoch 11 loss 0.15016856509772303 valid acc 15/16
 
Epoch 11 loss 0.28323600791196535 valid acc 15/16
 
Epoch 11 loss 0.28886369010882823 valid acc 16/16
 
Epoch 11 loss 0.1694711528694348 valid acc 16/16
 
Epoch 11 loss 0.5106606121797811 valid acc 15/16
 
Epoch 11 loss 0.42584857204323656 valid acc 15/16
 
Epoch 11 loss 0.2450386246407954 valid acc 15/16
 
Epoch 11 loss 0.24431914356730108 valid acc 15/16
 
Epoch 11 loss 0.81959975563673 valid acc 15/16
 
Epoch 11 loss 0.135072841396475 valid acc 15/16
 
Epoch 11 loss 0.4456596426368524 valid acc 15/16
 
Epoch 11 loss 0.7284037179641252 valid acc 15/16
 
Epoch 11 loss 0.7330078291790638 valid acc 15/16
 
Epoch 11 loss 0.410849955207059 valid acc 14/16
 
Epoch 11 loss 0.2943944840115214 valid acc 14/16
 
Epoch 12 loss 0.0043530154150722405 valid acc 15/16
 
Epoch 12 loss 0.5541580179266359 valid acc 15/16
 
Epoch 12 loss 0.8267604455316124 valid acc 15/16
 
Epoch 12 loss 0.27627370764266085 valid acc 15/16
 
Epoch 12 loss 0.11971647728438167 valid acc 15/16
 
Epoch 12 loss 0.09401047220139089 valid acc 15/16
 
Epoch 12 loss 0.38832261493239084 valid acc 15/16
 
Epoch 12 loss 0.5059407204022343 valid acc 15/16
 
Epoch 12 loss 0.39129720231696574 valid acc 15/16
 
Epoch 12 loss 0.4431267336158835 valid acc 15/16
 
Epoch 12 loss 0.2460541794211637 valid acc 15/16
 
Epoch 12 loss 1.2999033981223787 valid acc 15/16
 
Epoch 12 loss 0.5415567832391095 valid acc 15/16
 
Epoch 12 loss 0.4594929190286818 valid acc 15/16
 
Epoch 12 loss 0.30369852499851707 valid acc 16/16
 
Epoch 12 loss 0.2864295992460345 valid acc 14/16
 
Epoch 12 loss 0.4645220839234876 valid acc 15/16
 
Epoch 12 loss 0.4875464486538903 valid acc 14/16
 
Epoch 12 loss 0.3375467476658414 valid acc 14/16
 
Epoch 12 loss 0.19368501195830923 valid acc 15/16
 
Epoch 12 loss 0.30876958016222833 valid acc 15/16
 
Epoch 12 loss 0.18193957840675656 valid acc 15/16
 
Epoch 12 loss 0.030953882359015285 valid acc 15/16
 
Epoch 12 loss 0.34417184623206787 valid acc 13/16
 
Epoch 12 loss 0.41208306344020196 valid acc 14/16
 
Epoch 12 loss 0.4674387816202028 valid acc 15/16
 
Epoch 12 loss 0.3854424442955127 valid acc 15/16
 
Epoch 12 loss 0.18993004565542815 valid acc 16/16
 
Epoch 12 loss 0.11320798137028366 valid acc 15/16
 
Epoch 12 loss 0.16226637397205598 valid acc 15/16
 
Epoch 12 loss 0.3779707931058986 valid acc 15/16
 
Epoch 12 loss 0.4445835194594984 valid acc 15/16
 
Epoch 12 loss 0.12148902008489877 valid acc 15/16
 
Epoch 12 loss 0.8005926863022563 valid acc 15/16
 
Epoch 12 loss 0.9737761653206577 valid acc 15/16
 
Epoch 12 loss 0.24929942121369084 valid acc 15/16
 
Epoch 12 loss 0.32283145235406474 valid acc 15/16
 
Epoch 12 loss 0.3567564225748199 valid acc 15/16
 
Epoch 12 loss 0.7396991260883684 valid acc 16/16
 
Epoch 12 loss 0.2586920719600007 valid acc 14/16
 
Epoch 12 loss 0.18288996516646872 valid acc 15/16
 
Epoch 12 loss 0.42889990776692716 valid acc 15/16
 
Epoch 12 loss 0.6722940537536077 valid acc 15/16
 
Epoch 12 loss 0.1445571543029071 valid acc 14/16
 
Epoch 12 loss 0.6034118655643703 valid acc 16/16
 
Epoch 12 loss 0.1401592107763293 valid acc 16/16
 
Epoch 12 loss 0.4137700280488872 valid acc 15/16
 
Epoch 12 loss 0.3865433763299446 valid acc 16/16
 
Epoch 12 loss 0.26231203961677135 valid acc 16/16
 
Epoch 12 loss 0.5057332738213256 valid acc 16/16
 
Epoch 12 loss 0.8372816214982592 valid acc 14/16
 
Epoch 12 loss 0.6122918974608853 valid acc 16/16
 
Epoch 12 loss 0.3217729861497416 valid acc 15/16
 
Epoch 12 loss 0.13564545561260044 valid acc 15/16
 
Epoch 12 loss 0.6088579143469685 valid acc 15/16
 
Epoch 12 loss 0.48078698168453743 valid acc 15/16
 
Epoch 12 loss 0.5269897288978102 valid acc 14/16
 
Epoch 12 loss 0.16669434367477312 valid acc 13/16
 
Epoch 12 loss 0.4213468454824884 valid acc 15/16
 
Epoch 12 loss 0.27857198877618483 valid acc 15/16
 
Epoch 12 loss 0.30522846677846827 valid acc 15/16
 
Epoch 12 loss 0.28225499805782245 valid acc 14/16
 
Epoch 12 loss 0.3147816782827007 valid acc 16/16
 
Epoch 13 loss 0.005880927258460852 valid acc 15/16
 
Epoch 13 loss 0.33632770549188873 valid acc 14/16
 
Epoch 13 loss 0.7872001999075922 valid acc 16/16
 
Epoch 13 loss 0.35683312519881283 valid acc 14/16
 
Epoch 13 loss 0.20801303511709435 valid acc 15/16
 
Epoch 13 loss 0.5263710477343166 valid acc 14/16
 
Epoch 13 loss 0.44345308429091534 valid acc 15/16
 
Epoch 13 loss 0.5622367156525433 valid acc 14/16
 
Epoch 13 loss 0.7784312688429521 valid acc 14/16
 
Epoch 13 loss 0.3035780276361931 valid acc 14/16
 
Epoch 13 loss 0.1994964695092716 valid acc 16/16
 
Epoch 13 loss 0.9404347412397416 valid acc 15/16
 
Epoch 13 loss 0.4670555479236701 valid acc 15/16
 
Epoch 13 loss 0.3220373450027199 valid acc 15/16
 
Epoch 13 loss 0.1755048378486228 valid acc 15/16
 
Epoch 13 loss 0.1133773113628543 valid acc 15/16
 
Epoch 13 loss 0.5003372210272116 valid acc 14/16
 
Epoch 13 loss 1.0507555010581806 valid acc 15/16
 
Epoch 13 loss 0.45401380009618636 valid acc 14/16
 
Epoch 13 loss 0.421010613956336 valid acc 15/16
 
Epoch 13 loss 0.9224192520273761 valid acc 14/16
 
Epoch 13 loss 0.2278904425987107 valid acc 14/16
 
Epoch 13 loss 0.2110115183063057 valid acc 13/16
 
Epoch 13 loss 0.40854217091774575 valid acc 15/16
 
Epoch 13 loss 0.18837140223393187 valid acc 15/16
 
Epoch 13 loss 0.48352322553783567 valid acc 15/16
 
Epoch 13 loss 0.3512779194803409 valid acc 15/16
 
Epoch 13 loss 0.21217982900992932 valid acc 14/16
 
Epoch 13 loss 0.43648451265849264 valid acc 14/16
 
Epoch 13 loss 0.20578842330883707 valid acc 14/16
 
Epoch 13 loss 0.4858643274701735 valid acc 14/16
 
Epoch 13 loss 0.3973217695743896 valid acc 14/16
 
Epoch 13 loss 0.22154773291099464 valid acc 13/16
 
Epoch 13 loss 0.3425004298700211 valid acc 14/16
 
Epoch 13 loss 0.9932537758732541 valid acc 15/16
 
Epoch 13 loss 0.7706259652958278 valid acc 15/16
 
Epoch 13 loss 0.2682429948073779 valid acc 15/16
 
Epoch 13 loss 0.5919531343140755 valid acc 12/16
 
Epoch 13 loss 0.38415358097992836 valid acc 13/16
 
Epoch 13 loss 0.8952879766053907 valid acc 14/16
 
Epoch 13 loss 0.3249617973696162 valid acc 14/16
 
Epoch 13 loss 0.1613384580344569 valid acc 16/16
 
Epoch 13 loss 0.8129569185924181 valid acc 15/16
 
Epoch 13 loss 0.12297066816227482 valid acc 14/16
 
Epoch 13 loss 0.5890334316505151 valid acc 14/16
 
Epoch 13 loss 0.09494590529839421 valid acc 15/16
 
Epoch 13 loss 0.4974179824911816 valid acc 15/16
 
Epoch 13 loss 0.3315667178345206 valid acc 15/16
 
Epoch 13 loss 0.33715351098802926 valid acc 16/16
 
Epoch 13 loss 0.10389102598029801 valid acc 14/16
 
Epoch 13 loss 0.2102769045316509 valid acc 14/16
 
Epoch 13 loss 0.13853920301117428 valid acc 14/16
 
Epoch 13 loss 0.860455341491048 valid acc 14/16
 
Epoch 13 loss 0.39248768954091046 valid acc 15/16
 
Epoch 13 loss 0.35113591997693305 valid acc 15/16
 
Epoch 13 loss 0.6987437771206875 valid acc 14/16
 
Epoch 13 loss 1.0329484143945669 valid acc 15/16
 
Epoch 13 loss 0.056595133777103394 valid acc 15/16
 
Epoch 13 loss 0.31650264638338665 valid acc 14/16
 
Epoch 13 loss 0.9749025148908703 valid acc 15/16
 
Epoch 13 loss 0.2686125098962997 valid acc 15/16
 
Epoch 13 loss 0.14551120243702184 valid acc 15/16
 
Epoch 13 loss 0.15745422680017943 valid acc 12/16
 
Epoch 14 loss 0.00011888003385164081 valid acc 14/16
 
Epoch 14 loss 0.3097814747862745 valid acc 15/16
 
Epoch 14 loss 1.2851691868833093 valid acc 15/16
 
Epoch 14 loss 0.30568888304716346 valid acc 16/16
 
Epoch 14 loss 0.07967641546884294 valid acc 15/16
 
Epoch 14 loss 0.15661060194458618 valid acc 15/16
 
Epoch 14 loss 0.10921117290469769 valid acc 15/16
 
Epoch 14 loss 0.3792820886564373 valid acc 14/16
 
Epoch 14 loss 0.08560701355571754 valid acc 14/16
 
Epoch 14 loss 0.6527812003119158 valid acc 15/16
 
Epoch 14 loss 0.447022417518207 valid acc 15/16
 
Epoch 14 loss 0.5615734787907031 valid acc 15/16
 
Epoch 14 loss 0.6481638683385134 valid acc 16/16
 
Epoch 14 loss 0.4178721166649218 valid acc 15/16
 
Epoch 14 loss 0.4198102147917512 valid acc 15/16
 
Epoch 14 loss 0.1902059583395676 valid acc 15/16
 
Epoch 14 loss 0.5634047024846263 valid acc 15/16
 
Epoch 14 loss 0.43347746681257066 valid acc 14/16
 
Epoch 14 loss 0.3599387981424417 valid acc 15/16
 
Epoch 14 loss 0.330826243336415 valid acc 14/16
 
Epoch 14 loss 0.27514256559007066 valid acc 15/16
 
Epoch 14 loss 0.3086097423542027 valid acc 15/16
 
Epoch 14 loss 0.5127605804317519 valid acc 14/16
 
Epoch 14 loss 0.8352501322380346 valid acc 14/16
 
Epoch 14 loss 0.18515509775504896 valid acc 16/16
 
Epoch 14 loss 0.4359701529551449 valid acc 15/16
 
Epoch 14 loss 0.1551074122140137 valid acc 16/16
 
Epoch 14 loss 0.1241599577905883 valid acc 15/16
 
Epoch 14 loss 0.12244105983060789 valid acc 15/16
 
Epoch 14 loss 0.06521036572199762 valid acc 16/16
 
Epoch 14 loss 0.4542518725591621 valid acc 14/16
 
Epoch 14 loss 0.19305161739107718 valid acc 15/16
 
Epoch 14 loss 0.3083715299949854 valid acc 15/16
 
Epoch 14 loss 0.25989476540186596 valid acc 15/16
 
Epoch 14 loss 0.4651255202491149 valid acc 15/16
 
Epoch 14 loss 0.18067999297379025 valid acc 15/16
 
Epoch 14 loss 0.11029608333860125 valid acc 15/16
 
Epoch 14 loss 0.5673954906040075 valid acc 15/16
 
Epoch 14 loss 0.04748486141711504 valid acc 15/16
 
Epoch 14 loss 0.191437150961233 valid acc 15/16
 
Epoch 14 loss 0.20876847400944085 valid acc 15/16
 
Epoch 14 loss 0.053694028258923845 valid acc 15/16
 
Epoch 14 loss 0.2410813675223131 valid acc 15/16
 
Epoch 14 loss 0.6352294733669684 valid acc 15/16
 
Epoch 14 loss 0.6587261635462858 valid acc 15/16
 
Epoch 14 loss 0.21778591458738059 valid acc 15/16
 
Epoch 14 loss 0.27161953012479245 valid acc 15/16
 
Epoch 14 loss 1.0149018000944607 valid acc 16/16
 
Epoch 14 loss 0.09670012326131619 valid acc 14/16
 
Epoch 14 loss 0.5326744183315165 valid acc 15/16
 
Epoch 14 loss 0.3579380441876862 valid acc 15/16
 
Epoch 14 loss 0.5530132271073117 valid acc 15/16
 
Epoch 14 loss 0.7316661778787932 valid acc 15/16
 
Epoch 14 loss 0.04356934671905228 valid acc 14/16
 
Epoch 14 loss 0.43021199563658863 valid acc 15/16
 
Epoch 14 loss 0.1559404157208962 valid acc 15/16
 
Epoch 14 loss 0.5197450865468687 valid acc 15/16
 
Epoch 14 loss 0.27897469481219694 valid acc 15/16
 
Epoch 14 loss 0.5100651520607496 valid acc 14/16
 
Epoch 14 loss 0.07484762676214483 valid acc 14/16
 
Epoch 14 loss 0.31160331403494246 valid acc 15/16
 
Epoch 14 loss 0.08241413386659849 valid acc 15/16
 
Epoch 14 loss 0.37819376423482987 valid acc 15/16
 
Epoch 15 loss 0.18270975967438088 valid acc 15/16
 
Epoch 15 loss 0.18080854000007252 valid acc 15/16
 
Epoch 15 loss 0.2307377247428098 valid acc 16/16
 
Epoch 15 loss 0.7735421837841847 valid acc 16/16
 
Epoch 15 loss 0.18888149280481448 valid acc 15/16
 
Epoch 15 loss 0.8790278851086295 valid acc 15/16
 
Epoch 15 loss 0.47283763821320307 valid acc 13/16
 
Epoch 15 loss 1.0404040950982134 valid acc 15/16
 
Epoch 15 loss 0.35462152983457773 valid acc 15/16
 
Epoch 15 loss 0.2818379087596719 valid acc 14/16
 
Epoch 15 loss 0.11257941738631959 valid acc 16/16
 
Epoch 15 loss 0.7429108749313463 valid acc 15/16
 
Epoch 15 loss 0.16327002783285477 valid acc 14/16
 
Epoch 15 loss 1.0745394723969253 valid acc 15/16
 
Epoch 15 loss 0.5341165230751854 valid acc 15/16
 
Epoch 15 loss 0.47085170971843804 valid acc 15/16
 
Epoch 15 loss 0.5597637005391525 valid acc 15/16
 
Epoch 15 loss 0.34672884498256457 valid acc 15/16
 
Epoch 15 loss 0.8388167148774082 valid acc 15/16
 
Epoch 15 loss 0.35149684374951384 valid acc 15/16
 
Epoch 15 loss 0.4890690936097556 valid acc 14/16
 
Epoch 15 loss 0.29406830220824004 valid acc 15/16
 
Epoch 15 loss 0.01595570253423111 valid acc 15/16
 
Epoch 15 loss 0.36413427559975625 valid acc 13/16
 
Epoch 15 loss 0.32938195285886207 valid acc 14/16
 
Epoch 15 loss 0.2403240771777186 valid acc 14/16
 
Epoch 15 loss 0.40500054643055283 valid acc 15/16
 
Epoch 15 loss 0.039738046893628615 valid acc 15/16
 
Epoch 15 loss 0.07610153923903605 valid acc 15/16
 
Epoch 15 loss 0.043204230076402034 valid acc 15/16
 
Epoch 15 loss 0.33378855094816684 valid acc 15/16
 
Epoch 15 loss 0.2418828310050152 valid acc 15/16
 
Epoch 15 loss 0.129646061278752 valid acc 15/16
 
Epoch 15 loss 0.1206894125847154 valid acc 15/16
 
Epoch 15 loss 0.4173637781696872 valid acc 14/16
 
Epoch 15 loss 0.23486517008784005 valid acc 14/16
 
Epoch 15 loss 0.06810096127587777 valid acc 14/16
 
Epoch 15 loss 0.18626197882863826 valid acc 15/16
 
Epoch 15 loss 0.07622150744168374 valid acc 15/16
 
Epoch 15 loss 0.02248097664487904 valid acc 15/16
 
Epoch 15 loss 0.04794282369587084 valid acc 15/16
 
Epoch 15 loss 0.08112786716342653 valid acc 16/16
 
Epoch 15 loss 0.1467201146073377 valid acc 15/16
 
Epoch 15 loss 0.013302856360314903 valid acc 15/16
 
Epoch 15 loss 0.3272442153658758 valid acc 14/16
 
Epoch 15 loss 0.10928927104581335 valid acc 15/16
 
Epoch 15 loss 0.701821465544135 valid acc 15/16
 
Epoch 15 loss 0.6253505587625667 valid acc 15/16
 
Epoch 15 loss 0.05435465773240966 valid acc 15/16
 
Epoch 15 loss 0.4573549986181644 valid acc 15/16
 
Epoch 15 loss 0.1334716920157711 valid acc 15/16
 
Epoch 15 loss 0.6429340145122481 valid acc 16/16
 
Epoch 15 loss 0.45579857529173556 valid acc 15/16
 
Epoch 15 loss 0.6294827454629961 valid acc 16/16
 
Epoch 15 loss 0.09068044707147106 valid acc 16/16
 
Epoch 15 loss 0.11350785781796445 valid acc 16/16
 
Epoch 15 loss 0.5444443584846512 valid acc 15/16
 
Epoch 15 loss 0.3818495331753967 valid acc 15/16
 
Epoch 15 loss 0.2896760725302445 valid acc 15/16
 
Epoch 15 loss 0.514901610650017 valid acc 15/16
 
Epoch 15 loss 0.13687248281868025 valid acc 14/16
 
Epoch 15 loss 0.4002361843301394 valid acc 15/16
 
Epoch 15 loss 0.8206985245191699 valid acc 15/16
 
Epoch 16 loss 0.005084156548298396 valid acc 14/16
 
Epoch 16 loss 0.12483397584398576 valid acc 15/16
 
Epoch 16 loss 0.6379706833355214 valid acc 15/16
 
Epoch 16 loss 0.42116851841571146 valid acc 15/16
 
Epoch 16 loss 0.18499964774808922 valid acc 15/16
 
Epoch 16 loss 0.23459944975034291 valid acc 16/16
 
Epoch 16 loss 0.4141051136870915 valid acc 16/16
 
Epoch 16 loss 0.7397889557428073 valid acc 16/16
 
Epoch 16 loss 0.053850778853863315 valid acc 15/16
 
Epoch 16 loss 0.0708398917805825 valid acc 15/16
 
Epoch 16 loss 0.11255367653091702 valid acc 15/16
 
Epoch 16 loss 0.9757946614337705 valid acc 15/16
 
Epoch 16 loss 0.25045297905689434 valid acc 15/16
 
Epoch 16 loss 0.36139667683881277 valid acc 15/16
 
Epoch 16 loss 0.42938371799628167 valid acc 14/16
 
Epoch 16 loss 0.4871257682370078 valid acc 15/16
 
Epoch 16 loss 0.38263728499477134 valid acc 14/16
 
Epoch 16 loss 0.3985003968660852 valid acc 15/16
 
Epoch 16 loss 0.04750521683926015 valid acc 15/16
 
Epoch 16 loss 0.05937823095784346 valid acc 15/16
 
Epoch 16 loss 0.5072427023951323 valid acc 16/16
 
Epoch 16 loss 0.06154853691978019 valid acc 16/16
 
Epoch 16 loss 0.0038352779583984684 valid acc 15/16
 
Epoch 16 loss 0.23297082790497992 valid acc 16/16
 
Epoch 16 loss 0.20570672079928187 valid acc 15/16
 
Epoch 16 loss 0.4744043666150112 valid acc 16/16
 
Epoch 16 loss 0.8417486177977277 valid acc 16/16
 
Epoch 16 loss 0.02037329709844994 valid acc 16/16
 
Epoch 16 loss 0.9537402818825127 valid acc 15/16
 
Epoch 16 loss 0.3850705463244433 valid acc 14/16
 
Epoch 16 loss 0.015787713391740586 valid acc 15/16
 
Epoch 16 loss 0.044790839651277635 valid acc 15/16
 
Epoch 16 loss 0.13820655758595068 valid acc 14/16
 
Epoch 16 loss 0.40349383315514276 valid acc 15/16
 
Epoch 16 loss 0.7532009692742951 valid acc 15/16
 
Epoch 16 loss 0.05957499554839275 valid acc 15/16
 
Epoch 16 loss 0.14021953008716537 valid acc 15/16
 
Epoch 16 loss 0.3988015010080463 valid acc 15/16
 
Epoch 16 loss 0.14940503258128474 valid acc 16/16
 
Epoch 16 loss 0.1264949812663548 valid acc 15/16
 
Epoch 16 loss 0.19248787424798786 valid acc 15/16
 
Epoch 16 loss 0.03697959162971429 valid acc 16/16
 
Epoch 16 loss 0.15142005096934022 valid acc 16/16
 
Epoch 16 loss 0.7956312967066563 valid acc 14/16
 
Epoch 16 loss 0.14855813801646534 valid acc 15/16
 
Epoch 16 loss 0.009193018190244106 valid acc 16/16
 
Epoch 16 loss 0.20395330416270885 valid acc 15/16
 
Epoch 16 loss 0.9056512027184239 valid acc 15/16
 
Epoch 16 loss 0.7773205643535293 valid acc 15/16
 
Epoch 16 loss 0.13869135906180868 valid acc 15/16
 
Epoch 16 loss 0.1896567007481077 valid acc 14/16
 
Epoch 16 loss 0.1126649750469177 valid acc 15/16
 
Epoch 16 loss 0.14455499904678337 valid acc 15/16
 
Epoch 16 loss 0.8294352364674757 valid acc 14/16
 
Epoch 16 loss 0.7169526389548868 valid acc 15/16
 
Epoch 16 loss 0.21908783853882802 valid acc 15/16
 
Epoch 16 loss 0.4082856919377206 valid acc 14/16
 
Epoch 16 loss 0.10826388330369913 valid acc 14/16
 
Epoch 16 loss 0.7117518544129661 valid acc 14/16
 
Epoch 16 loss 1.3785755035402574 valid acc 15/16
 
Epoch 16 loss 0.3521633424861643 valid acc 15/16
 
Epoch 16 loss 0.6098588960657274 valid acc 14/16
 
Epoch 16 loss 0.5204951495339014 valid acc 14/16
 
Epoch 17 loss 0.0064462558195863525 valid acc 15/16
 
Epoch 17 loss 0.11565495010156944 valid acc 14/16
 
Epoch 17 loss 0.06677954205245362 valid acc 15/16
 
Epoch 17 loss 0.2963129463291169 valid acc 15/16
 
Epoch 17 loss 0.036609951233472014 valid acc 15/16
 
Epoch 17 loss 0.7779702851958348 valid acc 15/16
 
Epoch 17 loss 0.5959394357135785 valid acc 15/16
 
Epoch 17 loss 0.30289807133122526 valid acc 15/16
 
Epoch 17 loss 0.6619171175470102 valid acc 14/16
 
Epoch 17 loss 0.5254429361781965 valid acc 15/16
 
Epoch 17 loss 0.24763667431703218 valid acc 13/16
 
Epoch 17 loss 0.4108917332012148 valid acc 15/16
 
Epoch 17 loss 0.6674384807174609 valid acc 15/16
 
Epoch 17 loss 0.5201096523736579 valid acc 15/16
 
Epoch 17 loss 0.2500288913833056 valid acc 14/16
 
Epoch 17 loss 0.09650125653368069 valid acc 15/16
 
Epoch 17 loss 0.7122886955029384 valid acc 15/16
 
Epoch 17 loss 0.42709737822790766 valid acc 15/16
 
Epoch 17 loss 0.18138569290503367 valid acc 15/16
 
Epoch 17 loss 0.10603658420088946 valid acc 15/16
 
Epoch 17 loss 0.47103500091362205 valid acc 15/16
 
Epoch 17 loss 0.1839724653901249 valid acc 15/16
 
Epoch 17 loss 0.4463010860777341 valid acc 15/16
 
Epoch 17 loss 0.25145321086438366 valid acc 15/16
 
Epoch 17 loss 0.29190622076388023 valid acc 15/16
 
Epoch 17 loss 0.2904803406724563 valid acc 15/16
 
Epoch 17 loss 0.4647693368980636 valid acc 15/16
 
Epoch 17 loss 0.20690908662084675 valid acc 15/16
 
Epoch 17 loss 0.35691732037616186 valid acc 15/16
 
Epoch 17 loss 0.4729477457529461 valid acc 15/16
 
Epoch 17 loss 0.30686136596914765 valid acc 14/16
 
Epoch 17 loss 0.1552364120019008 valid acc 15/16
 
Epoch 17 loss 0.26939060292593026 valid acc 15/16
 
Epoch 17 loss 0.5118465647679052 valid acc 15/16
 
Epoch 17 loss 0.3280338030897405 valid acc 15/16
 
Epoch 17 loss 0.23651807617221746 valid acc 15/16
 
Epoch 17 loss 0.4181092492876745 valid acc 16/16
 
Epoch 17 loss 0.5301530793461299 valid acc 14/16
 
Epoch 17 loss 0.18195445382014644 valid acc 15/16
 
Epoch 17 loss 0.48399490868228573 valid acc 14/16
 
Epoch 17 loss 0.4467853522366921 valid acc 15/16
 
Epoch 17 loss 0.11866257506597055 valid acc 14/16
 
Epoch 17 loss 0.25785620539330123 valid acc 15/16
 
Epoch 17 loss 0.11485525565802074 valid acc 15/16
 
Epoch 17 loss 0.33785486757608224 valid acc 15/16
 
Epoch 17 loss 0.04658837922753214 valid acc 15/16
 
Epoch 17 loss 0.18072588652737015 valid acc 15/16
 
Epoch 17 loss 0.33786876990108267 valid acc 16/16
 
Epoch 17 loss 0.42876715550719224 valid acc 16/16
 
Epoch 17 loss 0.12365970013381711 valid acc 16/16
 
Epoch 17 loss 0.3614272025592512 valid acc 15/16
 
Epoch 17 loss 0.555636081461913 valid acc 14/16
 
Epoch 17 loss 0.11442461802669154 valid acc 15/16
 
Epoch 17 loss 0.06520242806685271 valid acc 14/16
 
Epoch 17 loss 0.4255763016729946 valid acc 14/16
 
Epoch 17 loss 0.15785474103679983 valid acc 15/16
 
Epoch 17 loss 0.1308777445257617 valid acc 15/16
 
Epoch 17 loss 0.42779978033632726 valid acc 15/16
 
Epoch 17 loss 0.16022031611718085 valid acc 14/16
 
Epoch 17 loss 0.1596239317035803 valid acc 15/16
 
Epoch 17 loss 0.0929144474150738 valid acc 15/16
 
Epoch 17 loss 0.3383914641744942 valid acc 15/16
 
Epoch 17 loss 0.2204276903215155 valid acc 15/16
 
Epoch 18 loss 0.0003048011557806455 valid acc 14/16
 
Epoch 18 loss 0.12657700645988282 valid acc 14/16
 
Epoch 18 loss 0.41436268732263537 valid acc 15/16
 
Epoch 18 loss 0.30649563028064813 valid acc 14/16
 
Epoch 18 loss 0.26378050872250397 valid acc 15/16
 
Epoch 18 loss 0.049853164352790066 valid acc 14/16
 
Epoch 18 loss 0.31866745078807013 valid acc 13/16
 
Epoch 18 loss 0.9394530789771354 valid acc 14/16
 
Epoch 18 loss 0.02213768663990726 valid acc 14/16
 
Epoch 18 loss 0.21153751878182017 valid acc 14/16
 
Epoch 18 loss 0.03470554894484063 valid acc 14/16
 
Epoch 18 loss 0.24043364656549876 valid acc 15/16
 
Epoch 18 loss 0.34140157392958026 valid acc 14/16
 
Epoch 18 loss 0.3961689513793431 valid acc 14/16
 
Epoch 18 loss 0.5041924322228494 valid acc 15/16
 
Epoch 18 loss 0.25876607224705106 valid acc 14/16
 
Epoch 18 loss 0.18427750221080766 valid acc 14/16
 
Epoch 18 loss 0.4536879838011142 valid acc 15/16
 
Epoch 18 loss 0.09829511256782701 valid acc 14/16
 
Epoch 18 loss 0.2353786786090747 valid acc 15/16
 
Epoch 18 loss 0.04502693624758893 valid acc 15/16
 
Epoch 18 loss 0.05702878587968532 valid acc 15/16
 
Epoch 18 loss 0.03387332596684525 valid acc 15/16
 
Epoch 18 loss 0.24913909389346695 valid acc 15/16
 
Epoch 18 loss 0.35544114016795186 valid acc 14/16
 
Epoch 18 loss 0.4038059357835396 valid acc 14/16
 
Epoch 18 loss 0.6331703380217127 valid acc 14/16
 
Epoch 18 loss 0.03791140152800243 valid acc 16/16
 
Epoch 18 loss 0.049191509677478346 valid acc 15/16
 
Epoch 18 loss 0.031728425870903285 valid acc 16/16
 
Epoch 18 loss 0.09038070771696194 valid acc 16/16
 
Epoch 18 loss 0.020723804371838732 valid acc 15/16
 
Epoch 18 loss 0.2346597897332774 valid acc 14/16
 
Epoch 18 loss 0.3913269680486874 valid acc 15/16
 
Epoch 18 loss 0.3500988342525444 valid acc 15/16
 
Epoch 18 loss 0.445458888057761 valid acc 15/16
 
Epoch 18 loss 0.6491937738366201 valid acc 15/16
 
Epoch 18 loss 0.5133660329369434 valid acc 16/16
 
Epoch 18 loss 0.21110140148203418 valid acc 13/16
 
Epoch 18 loss 0.3150424846756177 valid acc 15/16
 
Epoch 18 loss 0.06508533216125692 valid acc 15/16
 
Epoch 18 loss 0.12837721367707877 valid acc 15/16
 
Epoch 18 loss 0.25493829595834117 valid acc 13/16
 
Epoch 18 loss 0.21415239675772446 valid acc 14/16
 
Epoch 18 loss 0.2611730014584051 valid acc 14/16
 
Epoch 18 loss 0.21559679787240532 valid acc 15/16
 
Epoch 18 loss 0.44426461798757505 valid acc 15/16
 
Epoch 18 loss 0.3245263513643256 valid acc 15/16
 
Epoch 18 loss 0.05059914670354825 valid acc 15/16
 
Epoch 18 loss 0.03405203408712639 valid acc 15/16
 
Epoch 18 loss 0.20763181111501972 valid acc 15/16
 
Epoch 18 loss 0.11729589179568568 valid acc 15/16
 
Epoch 18 loss 0.5039696624058145 valid acc 15/16
 
Epoch 18 loss 0.21778828718037702 valid acc 15/16
 
Epoch 18 loss 0.2283058031813174 valid acc 15/16
 
Epoch 18 loss 0.14539570591343354 valid acc 15/16
 
Epoch 18 loss 0.5107520987844552 valid acc 15/16
 
Epoch 18 loss 0.11394230242868258 valid acc 15/16
 
Epoch 18 loss 0.5726116463655271 valid acc 14/16
 
Epoch 18 loss 0.35615218676918853 valid acc 15/16
 
Epoch 18 loss 0.01613057843673189 valid acc 15/16
 
Epoch 18 loss 0.08643745963240604 valid acc 14/16
 
Epoch 18 loss 0.22512535104459255 valid acc 14/16
 
Epoch 19 loss 0.00025992175384609024 valid acc 15/16
 
Epoch 19 loss 0.13973362554670862 valid acc 15/16
 
Epoch 19 loss 0.44695134456566254 valid acc 15/16
 
Epoch 19 loss 0.3884844155157124 valid acc 15/16
 
Epoch 19 loss 0.04402018125680791 valid acc 14/16
 
Epoch 19 loss 0.1144353652953831 valid acc 14/16
 
Epoch 19 loss 0.24224290348067004 valid acc 14/16
 
Epoch 19 loss 0.5281181859979625 valid acc 14/16
 
Epoch 19 loss 0.19721339834551255 valid acc 14/16
 
Epoch 19 loss 0.036646488041768115 valid acc 14/16
 
Epoch 19 loss 0.2460337589430157 valid acc 15/16
 
Epoch 19 loss 0.37237326156470724 valid acc 15/16
 
Epoch 19 loss 0.08619089467748071 valid acc 15/16
 
Epoch 19 loss 0.1136453031946615 valid acc 15/16
 
Epoch 19 loss 0.2195067654844243 valid acc 15/16
 
Epoch 19 loss 0.09432748598085738 valid acc 15/16
 
Epoch 19 loss 0.40800127365957384 valid acc 16/16
 
Epoch 19 loss 0.13062835861376282 valid acc 15/16
 
Epoch 19 loss 0.7387937661548403 valid acc 15/16
 
Epoch 19 loss 0.17260988513451594 valid acc 15/16
 
Epoch 19 loss 0.060735069506777054 valid acc 15/16
 
Epoch 19 loss 0.07541867309920142 valid acc 15/16
 
Epoch 19 loss 0.036110483297150775 valid acc 15/16
 
Epoch 19 loss 0.889430082680844 valid acc 15/16
 
Epoch 19 loss 0.1199924443552689 valid acc 15/16
 
Epoch 19 loss 0.2939690144453906 valid acc 15/16
 
Epoch 19 loss 0.43464583592736444 valid acc 14/16
 
Epoch 19 loss 0.20039704395197283 valid acc 14/16
 
Epoch 19 loss 0.2780857671709997 valid acc 14/16
 
Epoch 19 loss 0.08581132093331957 valid acc 14/16
 
Epoch 19 loss 0.5603658295164614 valid acc 14/16
 
Epoch 19 loss 0.28812632993981696 valid acc 14/16
 
Epoch 19 loss 0.15309058983880866 valid acc 13/16
 
Epoch 19 loss 0.7173242705344732 valid acc 15/16
 
Epoch 19 loss 0.24368668491462184 valid acc 15/16
 
Epoch 19 loss 0.1077335323331807 valid acc 15/16
 
Epoch 19 loss 0.1839371570304687 valid acc 14/16
 
Epoch 19 loss 0.0997352252905257 valid acc 15/16
 
Epoch 19 loss 0.129893841471869 valid acc 16/16
 
Epoch 19 loss 0.06956518323174707 valid acc 15/16
 
Epoch 19 loss 0.0334004626822775 valid acc 15/16
 
Epoch 19 loss 0.11653503838531773 valid acc 16/16
 
Epoch 19 loss 0.12777625459229844 valid acc 15/16
 
Epoch 19 loss 0.06331533775053008 valid acc 15/16
 
Epoch 19 loss 0.37056162144022564 valid acc 15/16
 
Epoch 19 loss 0.025670578902052137 valid acc 15/16
 
Epoch 19 loss 0.13592404316738416 valid acc 16/16
 
Epoch 19 loss 0.30440575135253434 valid acc 15/16
 
Epoch 19 loss 0.21171760734227776 valid acc 15/16
 
Epoch 19 loss 0.2554324835406175 valid acc 14/16
 
Epoch 19 loss 0.10049757218503202 valid acc 14/16
 
Epoch 19 loss 0.05391715449205528 valid acc 15/16
 
Epoch 19 loss 0.46996605468063274 valid acc 14/16
 
Epoch 19 loss 0.22488329987277733 valid acc 15/16
 
Epoch 19 loss 0.47608723613349646 valid acc 15/16
 
Epoch 19 loss 0.17746959519605873 valid acc 15/16
 
Epoch 19 loss 0.9041087482402352 valid acc 14/16
 
Epoch 19 loss 0.6282963229530127 valid acc 14/16
 
Epoch 19 loss 0.14398586362522717 valid acc 14/16
 
Epoch 19 loss 0.24286787100638776 valid acc 15/16
 
Epoch 19 loss 0.13370772283575355 valid acc 15/16
 
Epoch 19 loss 0.3662581558802109 valid acc 16/16
 
Epoch 19 loss 0.16359619446055412 valid acc 15/16
 
Epoch 20 loss 0.00016908292736210787 valid acc 15/16
 
Epoch 20 loss 0.30873370648494525 valid acc 15/16
 
Epoch 20 loss 0.2454140290416698 valid acc 15/16
 
Epoch 20 loss 0.17343930640199764 valid acc 15/16
 
Epoch 20 loss 0.024912271671631583 valid acc 15/16
 
Epoch 20 loss 0.05475420004728937 valid acc 15/16
 
Epoch 20 loss 0.1903318537074473 valid acc 15/16
 
Epoch 20 loss 0.25021519413509824 valid acc 14/16
 
Epoch 20 loss 0.7671943723106517 valid acc 15/16
 
Epoch 20 loss 0.4972049072315283 valid acc 16/16
 
Epoch 20 loss 0.05704006321073768 valid acc 16/16
 
Epoch 20 loss 0.16932720677468524 valid acc 15/16
 
Epoch 20 loss 0.2697292174422483 valid acc 16/16
 
Epoch 20 loss 0.07302256848916408 valid acc 16/16
 
Epoch 20 loss 0.033498735482310936 valid acc 15/16
 
Epoch 20 loss 0.24659968586131367 valid acc 16/16
 
Epoch 20 loss 0.15950742167671994 valid acc 15/16
 
Epoch 20 loss 0.07021915860883629 valid acc 15/16
 
Epoch 20 loss 0.09586859028642566 valid acc 16/16
 
Epoch 20 loss 0.4296528377515268 valid acc 15/16
 
Epoch 20 loss 0.0577798133310041 valid acc 15/16
 
Epoch 20 loss 0.065133510283415 valid acc 15/16
 
Epoch 20 loss 0.09796837842849998 valid acc 15/16
 
Epoch 20 loss 0.07346551454557607 valid acc 16/16
 
Epoch 20 loss 0.024574088270071088 valid acc 15/16
 
Epoch 20 loss 0.12862301432234113 valid acc 15/16
 
Epoch 20 loss 0.5067692614601458 valid acc 16/16
 
Epoch 20 loss 0.06394008664165873 valid acc 16/16
 
Epoch 20 loss 0.029427496460750255 valid acc 16/16
 
Epoch 20 loss 0.1669332689517332 valid acc 16/16
 
Epoch 20 loss 0.38518300885346324 valid acc 14/16
 
Epoch 20 loss 0.2045491118464252 valid acc 14/16
 
Epoch 20 loss 0.6138266018884287 valid acc 15/16
 
Epoch 20 loss 0.2458225744553152 valid acc 15/16
 
Epoch 20 loss 0.5480257308947212 valid acc 15/16
 
Epoch 20 loss 0.2676207552344189 valid acc 15/16
 
Epoch 20 loss 0.11735564816641011 valid acc 16/16
 
Epoch 20 loss 0.18845222349746538 valid acc 15/16
 
Epoch 20 loss 0.07519082117467045 valid acc 16/16
 
Epoch 20 loss 0.054309735030964694 valid acc 15/16
 
Epoch 20 loss 0.18011138045756786 valid acc 15/16
 
Epoch 20 loss 0.2956935596026425 valid acc 15/16
 
Epoch 20 loss 0.19452502427243112 valid acc 15/16
 
Epoch 20 loss 0.4047531414607879 valid acc 16/16
 
Epoch 20 loss 0.5269039234209903 valid acc 15/16
 
Epoch 20 loss 0.22296350561926345 valid acc 14/16
 
Epoch 20 loss 0.07704033675169167 valid acc 16/16
 
Epoch 20 loss 0.1181295563397908 valid acc 16/16
 
Epoch 20 loss 0.4030777484588794 valid acc 15/16
 
Epoch 20 loss 0.3873284475176305 valid acc 16/16
 
Epoch 20 loss 0.6688145938888451 valid acc 16/16
 
Epoch 20 loss 0.06057793413967271 valid acc 15/16
 
Epoch 20 loss 0.09306562526036552 valid acc 16/16
 
Epoch 20 loss 0.015333600940269731 valid acc 15/16
 
Epoch 20 loss 0.05119827671765884 valid acc 14/16
 
Epoch 20 loss 0.41843063792551793 valid acc 16/16
 
Epoch 20 loss 0.6987782518949075 valid acc 16/16
 
Epoch 20 loss 0.40870079647997737 valid acc 15/16
 
Epoch 20 loss 0.6089476494126372 valid acc 15/16
 
Epoch 20 loss 0.3121408026931855 valid acc 14/16
 
Epoch 20 loss 0.25438127935942073 valid acc 15/16
 
Epoch 20 loss 0.3354532655067464 valid acc 16/16
 
Epoch 20 loss 0.2379880738337364 valid acc 15/16
 
Epoch 21 loss 0.0052796054145182875 valid acc 16/16
 
Epoch 21 loss 0.040253045450361014 valid acc 14/16
 
Epoch 21 loss 0.2873593531608055 valid acc 15/16
 
Epoch 21 loss 0.1660152529409142 valid acc 16/16
 
Epoch 21 loss 0.6439695430123805 valid acc 16/16
 
Epoch 21 loss 0.11823058622188683 valid acc 15/16
 
Epoch 21 loss 0.47448480749732785 valid acc 15/16
 
Epoch 21 loss 0.24598629094742147 valid acc 15/16
 
Epoch 21 loss 0.08573630587055812 valid acc 16/16
 
Epoch 21 loss 0.16451060950705812 valid acc 16/16
 
Epoch 21 loss 0.035923949910645785 valid acc 15/16
 
Epoch 21 loss 0.21157280438153797 valid acc 16/16
 
Epoch 21 loss 0.2138238894577979 valid acc 16/16
 
Epoch 21 loss 0.5393644212073685 valid acc 16/16
 
Epoch 21 loss 0.31214534583900583 valid acc 16/16
 
Epoch 21 loss 0.8802787923601061 valid acc 15/16
 
Epoch 21 loss 0.3674003283854484 valid acc 15/16
 
Epoch 21 loss 0.12010358370023713 valid acc 15/16
 
Epoch 21 loss 0.1827243968480963 valid acc 14/16
 
Epoch 21 loss 0.03165375120683127 valid acc 16/16
 
Epoch 21 loss 0.21774286284837058 valid acc 15/16
 
Epoch 21 loss 0.0924433659693488 valid acc 16/16
 
Epoch 21 loss 0.20373973908693419 valid acc 15/16
 
Epoch 21 loss 0.4731888639425807 valid acc 15/16
 
Epoch 21 loss 0.09731391161767851 valid acc 14/16
 
Epoch 21 loss 0.2556890907227579 valid acc 14/16
 
Epoch 21 loss 0.185703569278957 valid acc 16/16
 
Epoch 21 loss 0.18159456523364298 valid acc 13/16
 
Epoch 21 loss 0.2928648880284863 valid acc 15/16
 
Epoch 21 loss 0.08042082699498011 valid acc 16/16
 
Epoch 21 loss 0.004897914816037078 valid acc 16/16
 
Epoch 21 loss 0.24563228920312033 valid acc 15/16
 
Epoch 21 loss 0.21094170719540878 valid acc 16/16
 
Epoch 21 loss 0.798972979041812 valid acc 15/16
 
Epoch 21 loss 0.35042175962767713 valid acc 15/16
 
Epoch 21 loss 0.11978961958435676 valid acc 16/16
 
Epoch 21 loss 0.11310208920096555 valid acc 16/16
 
Epoch 21 loss 0.049650069783083055 valid acc 15/16
 
Epoch 21 loss 0.3362933767341953 valid acc 16/16
 
Epoch 21 loss 0.019795975143460685 valid acc 15/16
 
Epoch 21 loss 0.03554263505671797 valid acc 16/16
 
Epoch 21 loss 0.014984622992180778 valid acc 16/16
 
Epoch 21 loss 0.09301581626138661 valid acc 15/16
 
Epoch 21 loss 0.06501506862543 valid acc 16/16
 
Epoch 21 loss 0.08817702935636369 valid acc 15/16
 
Epoch 21 loss 0.11145967809835505 valid acc 15/16
 
Epoch 21 loss 0.3571563584909404 valid acc 15/16
 
Epoch 21 loss 0.43974811431093613 valid acc 14/16
 
Epoch 21 loss 0.19931497982160662 valid acc 15/16
 
Epoch 21 loss 0.7502061111527407 valid acc 16/16
 
Epoch 21 loss 0.3613062808101393 valid acc 14/16
 
Epoch 21 loss 0.15499421020413146 valid acc 16/16
 
Epoch 21 loss 0.09685150403631788 valid acc 14/16
 
Epoch 21 loss 0.057737536583981106 valid acc 15/16
 
Epoch 21 loss 0.20339486602150472 valid acc 15/16
 
Epoch 21 loss 0.10995488648224337 valid acc 14/16
 
Epoch 21 loss 0.11708963575810469 valid acc 15/16
 
Epoch 21 loss 0.06051331659668041 valid acc 15/16
 
Epoch 21 loss 0.028536247579244223 valid acc 15/16
 
Epoch 21 loss 0.33275752845082923 valid acc 15/16
 
Epoch 21 loss 0.461959853217809 valid acc 14/16
 
Epoch 21 loss 0.14731365440631677 valid acc 15/16
 
Epoch 21 loss 0.09749342207296495 valid acc 15/16
 
Epoch 22 loss 0.004009232405793772 valid acc 16/16
 
Epoch 22 loss 0.3028883437225107 valid acc 14/16
 
Epoch 22 loss 0.4210895225505011 valid acc 16/16
 
Epoch 22 loss 0.08285128517931463 valid acc 15/16
 
Epoch 22 loss 0.01517583785464168 valid acc 16/16
 
Epoch 22 loss 0.03250836136510939 valid acc 15/16
 
Epoch 22 loss 0.9636431369393519 valid acc 15/16
 
Epoch 22 loss 0.3898467171826591 valid acc 15/16
 
Epoch 22 loss 0.16797203921509632 valid acc 15/16
 
Epoch 22 loss 0.31965235258992813 valid acc 15/16
 
Epoch 22 loss 0.31804709285379695 valid acc 15/16
 
Epoch 22 loss 0.535492541870633 valid acc 15/16
 
Epoch 22 loss 0.519232505215701 valid acc 14/16
 
Epoch 22 loss 0.24590162701734075 valid acc 14/16
 
Epoch 22 loss 0.0665819308385931 valid acc 15/16
 
Epoch 22 loss 0.31733688956340117 valid acc 15/16
 
Epoch 22 loss 0.35468825510389157 valid acc 14/16
 
Epoch 22 loss 0.5198009636980986 valid acc 15/16
 
Epoch 22 loss 0.2581622792759343 valid acc 15/16
 
Epoch 22 loss 0.021064755484481265 valid acc 15/16
 
Epoch 22 loss 0.677452981135837 valid acc 13/16
 
Epoch 22 loss 0.5150148609099268 valid acc 16/16
 
Epoch 22 loss 0.5875868762031902 valid acc 13/16
 
Epoch 22 loss 0.10813231245563995 valid acc 12/16
 
Epoch 22 loss 0.25683116157155206 valid acc 13/16
 
Epoch 22 loss 0.3498170934960121 valid acc 14/16
 
Epoch 22 loss 0.07035138201803398 valid acc 15/16
 
Epoch 22 loss 0.4054502582172788 valid acc 14/16
 
Epoch 22 loss 0.028546744420525107 valid acc 14/16
 
Epoch 22 loss 0.044113043032426386 valid acc 14/16
 
Epoch 22 loss 0.5990373885223692 valid acc 15/16
 
Epoch 22 loss 0.10764867390373732 valid acc 15/16
 
Epoch 22 loss 0.19399124335666873 valid acc 14/16
 
Epoch 22 loss 0.32349732771961787 valid acc 14/16
 
Epoch 22 loss 0.19535171084110722 valid acc 15/16
 
Epoch 22 loss 0.3128640119191936 valid acc 15/16
 
Epoch 22 loss 0.144939235335709 valid acc 15/16
 
Epoch 22 loss 0.7955347814091062 valid acc 15/16
 
Epoch 22 loss 0.3699651437121263 valid acc 15/16
 
Epoch 22 loss 0.24127700135196292 valid acc 14/16
 
Epoch 22 loss 0.03382914673845627 valid acc 14/16
 
Epoch 22 loss 0.12519177835636916 valid acc 14/16
 
Epoch 22 loss 0.03318121931257241 valid acc 14/16
 
Epoch 22 loss 0.24851211194695422 valid acc 14/16
 
Epoch 22 loss 0.014108613631616269 valid acc 14/16
 
Epoch 22 loss 0.012737850025233805 valid acc 16/16
 
Epoch 22 loss 0.08911632707662191 valid acc 14/16
 
Epoch 22 loss 0.32448651592096417 valid acc 15/16
 
Epoch 22 loss 0.1780462633469564 valid acc 15/16
 
Epoch 22 loss 0.0482383370184425 valid acc 15/16
 
Epoch 22 loss 0.048392661002019596 valid acc 14/16
 
Epoch 22 loss 0.31472144670206914 valid acc 14/16
 
Epoch 22 loss 0.8207695544940175 valid acc 15/16
 
Epoch 22 loss 0.20428499546315115 valid acc 15/16
 
Epoch 22 loss 0.4281022145258682 valid acc 15/16
 
Epoch 22 loss 0.31304475609580396 valid acc 15/16
 
Epoch 22 loss 0.2929919552523832 valid acc 14/16
 
Epoch 22 loss 0.509474656296363 valid acc 14/16
 
Epoch 22 loss 0.15106294406224482 valid acc 15/16
 
Epoch 22 loss 0.09008254297456442 valid acc 15/16
 
Epoch 22 loss 0.0772527294661629 valid acc 15/16
 
Epoch 22 loss 0.2003916842481368 valid acc 14/16
 
Epoch 22 loss 0.36749619392880545 valid acc 15/16
 
Epoch 23 loss 0.002947592798586761 valid acc 15/16
 
Epoch 23 loss 0.1378806810037038 valid acc 15/16
 
Epoch 23 loss 0.04980703960907508 valid acc 14/16
 
Epoch 23 loss 0.12892684460272022 valid acc 15/16
 
Epoch 23 loss 0.23927191231462563 valid acc 14/16
 
Epoch 23 loss 0.10299596966610369 valid acc 15/16
 
Epoch 23 loss 0.10174639996902343 valid acc 15/16
 
Epoch 23 loss 0.19760158538103795 valid acc 15/16
 
Epoch 23 loss 0.053284388025318014 valid acc 15/16
 
Epoch 23 loss 0.1380768649384256 valid acc 14/16
 
Epoch 23 loss 0.06809435852626736 valid acc 15/16
 
Epoch 23 loss 0.35976608943839694 valid acc 14/16
 
Epoch 23 loss 0.671994378170378 valid acc 15/16
 
Epoch 23 loss 0.043211045711474376 valid acc 14/16
 
Epoch 23 loss 0.49998911241089483 valid acc 16/16
 
Epoch 23 loss 0.10644381768085105 valid acc 16/16
 
Epoch 23 loss 0.35441064140782136 valid acc 15/16
 
Epoch 23 loss 0.22150945103247127 valid acc 15/16
 
Epoch 23 loss 0.2944758153989533 valid acc 15/16
 
Epoch 23 loss 0.20760412748216567 valid acc 15/16
 
Epoch 23 loss 0.15333692570313132 valid acc 15/16
 
Epoch 23 loss 0.05208605671141896 valid acc 15/16
 
Epoch 23 loss 0.006853801307489542 valid acc 15/16
 
Epoch 23 loss 0.11224856698037927 valid acc 15/16
 
Epoch 23 loss 0.11660099349964619 valid acc 14/16
 
Epoch 23 loss 0.22445658679041658 valid acc 15/16
 
Epoch 23 loss 0.11309871690366675 valid acc 14/16
 
Epoch 23 loss 0.29286076343711315 valid acc 16/16
 
Epoch 23 loss 0.9792162227416623 valid acc 15/16
 
Epoch 23 loss 0.31923972965339187 valid acc 15/16
 
Epoch 23 loss 0.16000970770672154 valid acc 14/16
 
Epoch 23 loss 0.020944898064585157 valid acc 15/16
 
Epoch 23 loss 0.21131624603386395 valid acc 15/16
 
Epoch 23 loss 0.08699141147580859 valid acc 14/16
 
Epoch 23 loss 0.554929860694008 valid acc 15/16
 
Epoch 23 loss 0.13893351793029785 valid acc 15/16
 
Epoch 23 loss 0.20017171139337692 valid acc 15/16
 
Epoch 23 loss 0.13979255243332606 valid acc 15/16
 
Epoch 23 loss 0.16166378678095053 valid acc 16/16
 
Epoch 23 loss 0.23051691107453373 valid acc 15/16
 
Epoch 23 loss 0.1296489339679534 valid acc 15/16
 
Epoch 23 loss 0.022458736411870323 valid acc 15/16
 
Epoch 23 loss 0.19061379001455805 valid acc 14/16
 
Epoch 23 loss 0.015396333434981277 valid acc 14/16
 
Epoch 23 loss 0.3906351843899691 valid acc 16/16
 
Epoch 23 loss 0.18179834686624916 valid acc 15/16
 
Epoch 23 loss 0.48491606223976647 valid acc 14/16
 
Epoch 23 loss 0.3479067046178164 valid acc 13/16
 
Epoch 23 loss 0.2661257637241456 valid acc 15/16
 
Epoch 23 loss 0.14496920567187585 valid acc 14/16
 
Epoch 23 loss 0.46387335587377326 valid acc 15/16
 
Epoch 23 loss 0.3409171081243918 valid acc 15/16
 
Epoch 23 loss 0.29222584042075767 valid acc 14/16
 
Epoch 23 loss 0.12433079369619125 valid acc 15/16
 
Epoch 23 loss 0.07492036844325416 valid acc 15/16
 
Epoch 23 loss 0.30939224743930005 valid acc 15/16
 
Epoch 23 loss 0.2641357430044727 valid acc 15/16
 
Epoch 23 loss 0.005811884030207017 valid acc 15/16
 
Epoch 23 loss 0.18671613712647958 valid acc 15/16
 
Epoch 23 loss 0.10403203536007168 valid acc 14/16
 
Epoch 23 loss 0.022135649138797578 valid acc 14/16
 
Epoch 23 loss 0.016730471152791138 valid acc 15/16
 
Epoch 23 loss 0.0802526102493682 valid acc 14/16
 
Epoch 24 loss 0.028796591316716892 valid acc 14/16
 
Epoch 24 loss 0.11448326721050517 valid acc 14/16
 
Epoch 24 loss 0.1468483895345898 valid acc 15/16
 
Epoch 24 loss 0.16164665340356704 valid acc 15/16
 
Epoch 24 loss 0.07058662481523933 valid acc 15/16
 
Epoch 24 loss 0.1966850674084812 valid acc 16/16
 
Epoch 24 loss 0.3227714529458274 valid acc 14/16
 
Epoch 24 loss 1.4172417748015544 valid acc 14/16
 
Epoch 24 loss 1.0006475637922305 valid acc 15/16
 
Epoch 24 loss 0.13347628611783965 valid acc 15/16
 
Epoch 24 loss 0.28542663130001716 valid acc 14/16
 
Epoch 24 loss 0.35309564312298686 valid acc 15/16
 
Epoch 24 loss 0.19508477416662076 valid acc 14/16
 
Epoch 24 loss 0.45919812624215206 valid acc 14/16
 
Epoch 24 loss 0.0818953112961614 valid acc 15/16
 
Epoch 24 loss 0.46885641067733913 valid acc 16/16
 
Epoch 24 loss 0.43354493628790475 valid acc 15/16
 
Epoch 24 loss 0.3115643251185335 valid acc 15/16
 
Epoch 24 loss 0.05517275742310893 valid acc 16/16
 
Epoch 24 loss 0.3241736844872656 valid acc 15/16
 
Epoch 24 loss 0.7333415592521995 valid acc 15/16
 
Epoch 24 loss 0.3615235800122408 valid acc 15/16
 
Epoch 24 loss 0.012167166822709349 valid acc 15/16
 
Epoch 24 loss 0.04565981023787782 valid acc 14/16
 
Epoch 24 loss 0.15469373156075136 valid acc 14/16
 
Epoch 24 loss 0.1096494321992857 valid acc 14/16
 
Epoch 24 loss 0.07977897859375696 valid acc 14/16
 
Epoch 24 loss 0.12473894861134138 valid acc 14/16
 
Epoch 24 loss 0.38535689843991183 valid acc 14/16
 
Epoch 24 loss 0.4216408004915605 valid acc 15/16
 
Epoch 24 loss 0.19829182820729058 valid acc 15/16
 
Epoch 24 loss 0.577477010137279 valid acc 13/16
 
Epoch 24 loss 0.43171132692850217 valid acc 14/16
 
Epoch 24 loss 0.792342862481666 valid acc 15/16
 
Epoch 24 loss 0.2556036568036165 valid acc 14/16
 
Epoch 24 loss 0.18159242291342276 valid acc 15/16
 
Epoch 24 loss 0.07148906360137061 valid acc 16/16
 
Epoch 24 loss 0.07556314564278993 valid acc 16/16
 
Epoch 24 loss 0.0410320981663196 valid acc 14/16
 
Epoch 24 loss 0.22861186618065923 valid acc 16/16
 
Epoch 24 loss 0.0013744157171210582 valid acc 16/16
 
Epoch 24 loss 0.07087444496880087 valid acc 16/16
 
Epoch 24 loss 0.2502484964536008 valid acc 13/16
 
Epoch 24 loss 0.13991863422169204 valid acc 14/16
 
Epoch 24 loss 0.27114423904678825 valid acc 14/16
 
Epoch 24 loss 0.23913608402428987 valid acc 16/16
 
Epoch 24 loss 0.45805585404901084 valid acc 15/16
 
Epoch 24 loss 0.2240357189962776 valid acc 15/16
 
Epoch 24 loss 0.15768778498691077 valid acc 14/16
 
Epoch 24 loss 0.01637072220470416 valid acc 15/16
 
Epoch 24 loss 0.18708889112064325 valid acc 14/16
 
Epoch 24 loss 0.2781266031169282 valid acc 15/16
 
Epoch 24 loss 1.2355588974712932 valid acc 15/16
 
Epoch 24 loss 0.09554497178919116 valid acc 15/16
 
Epoch 24 loss 0.10364483306220389 valid acc 15/16
 
Epoch 24 loss 0.41305258065258954 valid acc 15/16
 
Epoch 24 loss 0.3623723857966288 valid acc 15/16
 
Epoch 24 loss 0.047071826651084614 valid acc 14/16
 
Epoch 24 loss 0.1741378727075693 valid acc 15/16
 
Epoch 24 loss 0.29344170425900473 valid acc 15/16
 
Epoch 24 loss 0.2580528101857574 valid acc 15/16
 
Epoch 24 loss 0.4318490142549153 valid acc 16/16
 
Epoch 24 loss 0.29599270410608225 valid acc 14/16
 
Epoch 25 loss 0.0025149826780291214 valid acc 14/16
 
Epoch 25 loss 0.16230747789990596 valid acc 13/16
 
Epoch 25 loss 0.14835413141163142 valid acc 15/16
 
Epoch 25 loss 0.35323339962846567 valid acc 15/16
 
Epoch 25 loss 0.03237522756690261 valid acc 15/16
 
Epoch 25 loss 0.07236915938471422 valid acc 15/16
 
Epoch 25 loss 0.07683092148912637 valid acc 14/16
 
Epoch 25 loss 0.16240998205437793 valid acc 14/16
 
Epoch 25 loss 0.40353852836118953 valid acc 15/16
 
Epoch 25 loss 0.4836409418206619 valid acc 13/16
 
Epoch 25 loss 0.030487662840089313 valid acc 15/16
 
Epoch 25 loss 0.3372081825405298 valid acc 15/16
 
Epoch 25 loss 0.20974763537877553 valid acc 15/16
 
Epoch 25 loss 0.4269958260630718 valid acc 14/16
 
Epoch 25 loss 0.08810997002305856 valid acc 14/16
 
Epoch 25 loss 0.09521463434151539 valid acc 14/16
 
Epoch 25 loss 0.17362572660073675 valid acc 15/16
 
Epoch 25 loss 0.14753472955039848 valid acc 14/16
 
Epoch 25 loss 0.4279477674435385 valid acc 15/16
 
Epoch 25 loss 0.09369661735817403 valid acc 14/16
 
Epoch 25 loss 0.2480906776985581 valid acc 14/16
 
Epoch 25 loss 0.2196246859130677 valid acc 15/16
 
Epoch 25 loss 0.003944295013848136 valid acc 15/16
 
Epoch 25 loss 0.09041482678027239 valid acc 15/16
 
Epoch 25 loss 0.019180195179101814 valid acc 14/16
 
Epoch 25 loss 0.39024558328707937 valid acc 14/16
 
Epoch 25 loss 0.3486090745797715 valid acc 16/16
 
Epoch 25 loss 0.1339298998083965 valid acc 16/16
 
Epoch 25 loss 0.02116570597862248 valid acc 16/16
 
Epoch 25 loss 0.1203221744845571 valid acc 15/16
 
Epoch 25 loss 0.007825182646015704 valid acc 16/16
 
Epoch 25 loss 0.19259307308913248 valid acc 14/16
 
Epoch 25 loss 0.04993574073166844 valid acc 14/16
 
Epoch 25 loss 0.6115386964158269 valid acc 15/16
 
Epoch 25 loss 0.2478695865486945 valid acc 14/16
 
Epoch 25 loss 0.03894125375930543 valid acc 15/16
 
Epoch 25 loss 0.15862025561223753 valid acc 14/16
 
Epoch 25 loss 0.2675392064500344 valid acc 15/16
 
Epoch 25 loss 0.23651611447459417 valid acc 15/16
 
Epoch 25 loss 0.0729550140757798 valid acc 14/16
 
Epoch 25 loss 0.06888387458591863 valid acc 15/16
 
Epoch 25 loss 0.21193791145318872 valid acc 15/16
 
Epoch 25 loss 0.45787037731409874 valid acc 15/16
 
Epoch 25 loss 0.06662822184409771 valid acc 14/16
 
Epoch 25 loss 0.31903549426156286 valid acc 15/16
 
Epoch 25 loss 0.012829372914596713 valid acc 16/16
 
Epoch 25 loss 0.15798367596101323 valid acc 15/16
 
Epoch 25 loss 0.22046047878876954 valid acc 15/16
 
Epoch 25 loss 0.11276465883372125 valid acc 15/16
 
Epoch 25 loss 0.09792011307344241 valid acc 15/16
 
Epoch 25 loss 0.20086475658440456 valid acc 16/16
 
Epoch 25 loss 0.07313923620832165 valid acc 15/16
 
Epoch 25 loss 0.06510957426131729 valid acc 15/16
 
Epoch 25 loss 0.014227232573313146 valid acc 15/16
 
Epoch 25 loss 0.07492929898032781 valid acc 15/16
 
Epoch 25 loss 0.23354321980040385 valid acc 15/16
 
Epoch 25 loss 0.2713577801348595 valid acc 14/16
 
Epoch 25 loss 0.004862177775425651 valid acc 14/16
 
Epoch 25 loss 0.23064351986729276 valid acc 14/16
 
Epoch 25 loss 0.22336455008882003 valid acc 14/16
 
Epoch 25 loss 0.0851919412167975 valid acc 14/16
 
Epoch 25 loss 0.04629889212401579 valid acc 15/16
 
Epoch 25 loss 0.2909221579227738 valid acc 15/16