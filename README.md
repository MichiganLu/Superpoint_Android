# Superpoint Android and X64 Implementation
![results](x64_ipool_0.008_0.77_top1k.jpg)

**Supporting NCNN and SNPE implementation, runs on Ubuntu and Android** (tested for Ubuntu18 and Android24).

**NO EXTRA DEPENDENCY!!!**  Just compile and run.

## Running on Ubuntu x64 machine
I will give an example of how to run with NCNN, it is similar procedure to run with SNPE.
```
git clone git@github.com:MichiganLu/Superpoint_android.git

cd Superpoint_android/superpoint_NCNN

bash compile.sh

cd build/x64/bin/

./NCNN_superpoint

```
Simple as that, smooth and automatic.

## Running on Android machine
First change the android NDK path in [compile.sh](superpoint_NCNN/compile.sh) for NCNN or [compile.sh](superpoint_SNPE/compile.sh) for SNPE. Then compile with the -f android64 flag, just like below
```
bash compile.sh -f android64
```
Then run it on Android.
