# occamsworkspace
ben rayfield's research workspace, that runs when you doubleclick startInWindows.bat and can be set up in Eclipse etc, still some projects to merge in, trying to keep things simple (occams razor)

To run it in win10 (possibly other versions of windows, TODO linux),
download zip from https://github.com/benrayfield/occamsworkspace/
unzip the openjdk-11.0.2_windows-x64_bin.7z.001 (auto includes .002) into selfContained\data\lib\openjdk-11.0.2_windows-x64_bin
double click selfContained/startInWindows.bat

(planning for startInLinux.sh to also work, but will have to include lwjgl2 linux version first)

This includes GPU code, that should automatically work.


A self contained workspace you can copy/paste to other computers, OS, andOr dirs, run startIn* file, and a window appears on screen, server starts, etc, automatically. Further options are available within the program.

TODO parts will include: jython, java, occamsfuncer, jsoundcard, occamserver, lazycl, a fork of recurrentjava thats gpu optimized, a selfModifying IDE, a fork of listweb mindmap software (which can handle up to a million manually written objects in the web of things connected to things, unlike other mindmaps which are tree based and can barely handle 1000 as it gets visually cluttered).


Copied this from humanainetneural readme 2021-3-26 which is part of this software[
# HumanAiNetNeural
just the neuralnet parts of HumanAiNet, which (TODO) will include learnloop/RBM, feedforward, and opencl optimized RecurrentJava LSTM and GRU

<nobr><img src=https://github.com/benrayfield/HumanAiNet/raw/master/data/website/rbm2018-4.png height=250>
<img src=https://github.com/benrayfield/HumanAiNet/raw/master/data/website/rbm2018-5.png height=250></nobr>
]
