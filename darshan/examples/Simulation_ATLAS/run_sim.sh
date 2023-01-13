#!/bin/bash
localdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# export DARSHAN_BASE_DIR=/software/centos7/soft/darshan/3.2.1/
# export DARSHAN_BASE_DIR=/lcrc/group/ATLAS/users/rwang/Argonne_computing/PPS-CCE/darshan/build_darshan/darshan-3.4.0-pre1
export DARSHAN_BASE_DIR=/lcrc/group/ATLAS/users/rwang/Argonne_computing/PPS-CCE/darshan/build_darshan/dev-fork-child-issue786
export DARSHAN_LOG_DIR=$($DARSHAN_BASE_DIR/bin/darshan-config --log-path)

athena=$1
nevents_per_proc=$2 #5
nproc=$3 #32
simulator='FullG4' #'ATLFASTII'
sharedWriter=$4 #False
proc=$5
workdir=Athena_23_0_5/$proc

case ${proc} in
"Jpsimumu")
    inputfile='/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ISF_Validation/mc21_13p6TeV.801164.P8B_A14_CTEQ6L1_bb_Jpsi1S_mu6mu4.merge.EVNT.e8453_e8455.29328730._000257.pool.root.1'
;;
"ttbar")
    inputfile="/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/SimCoreTests/valid1.410000.PowhegPythiaEvtGen_P2012_ttbar_hdamp172p5_nonallhad.evgen.EVNT.e4993.EVNT.08166201._000012.pool.root.1"
esac
# export DARSHAN_DEFAULT_NPROCS=7
nevents=$(($nproc * $nevents_per_proc))
subfolder=$(date +'%Y/%m/%d')
logfolder=$DARSHAN_LOG_DIR/${subfolder//"/0"/"/"}
# PROG=athena_sim
# export DARSHAN_LOGFILE=$DARSHAN_LOG_DIR/$(date +'%y/%m/%d')/${PROG}.darshan
# rm -f ${DARSHAN_LOGFILE}
export DARSHAN_ENABLE_NONMPI=1
cp $localdir/env.conf workdir/$workdir/
export DARSHAN_CONFIG_PATH=$localdir/env.conf 
export DARSHAN_DUMP_CONFIG=1
# export DARSHAN_JOBID=PBS_JOBID

case ${athena} in
"athena")
    #--- athena ---
    nproc=1 && workdir=workdir/$workdir/athena_${simulator}_ttbar_${nproc}_${nevents_per_proc}
    echo "working in $workdir"
    (mkdir -p $workdir && cd $workdir && rm -rf * && Sim_tf.py --conditionsTag 'default:OFLCOND-MC16-SDR-14' --physicsList 'FTFP_BERT_ATL' --truthStrategy 'MC15aPlus' --simulator $simulator --postInclude 'default:PyJobTransforms/UseFrontier.py' --DataRunNumber '284500' --geometryVersion 'default:ATLAS-R2-2016-01-00-01' --inputEVNTFile $inputfile --outputHITSFile "test.HITS.pool.root" --maxEvents ${nevents_per_proc} --imf False --athenaopts=' --stdcmalloc --preloadlib=$DARSHAN_BASE_DIR/lib/libdarshan.so' 2>&1 |tee $localdir/$workdir.log)
;;
"athenaMP")
    #--- athenaMP ---
    workdir=workdir/$workdir/athenaMP_${simulator}_ttbar_${nproc}_${nevents_per_proc}_${sharedWriter}
    echo "working in $workdir"
    (mkdir -p $workdir && cd $workdir && rm -rf * && ATHENA_CORE_NUMBER=$nproc Sim_tf.py --conditionsTag 'default:OFLCOND-MC16-SDR-14' --physicsList 'FTFP_BERT_ATL' --truthStrategy 'MC15aPlus' --simulator $simulator --postInclude 'default:PyJobTransforms/UseFrontier.py' --DataRunNumber '284500' --geometryVersion 'default:ATLAS-R2-2016-01-00-01' --inputEVNTFile $inputfile --outputHITSFile "test.HITS.pool.root" --maxEvents ${nevents} --imf False --sharedWriter ${sharedWriter} --multiprocess True --athenaMPStrategy 'SharedQueue' --athenaopts=' --stdcmalloc --preloadlib=$DARSHAN_BASE_DIR/lib/libdarshan.so' 2>&1 |tee $localdir/$workdir.log)
;;
"athenaMT")
    #--- athenaMT ---
    workdir=workdir/$workdir/athenaMT_${simulator}_ttbar_${nproc}_${nevents_per_proc}
    echo "working in $workdir"
    (mkdir -p $workdir && cd $workdir && rm -rf * && ATHENA_CORE_NUMBER=${nproc} Sim_tf.py --conditionsTag 'default:OFLCOND-MC16-SDR-14' --physicsList 'FTFP_BERT_ATL' --truthStrategy 'MC15aPlus' --simulator ${simulator}MT --postInclude 'default:PyJobTransforms/UseFrontier.py' --DataRunNumber '284500' --geometryVersion 'default:ATLAS-R2-2016-01-00-01' --inputEVNTFile $inputfile --outputHITSFile "test.HITS.pool.root" --maxEvents ${nevents} --imf False --multithreaded True --athenaopts=' --stdcmalloc --preloadlib=$DARSHAN_BASE_DIR/lib/libdarshan.so' 2>&1 |tee $localdir/${workdir}.log)
esac

# (cd FullG4_ttbar && athena.py --preloadlib=$DARSHAN_BASE_DIR/lib/libdarshan.so runargs.EVNTtoHITS.py SimuJobTransforms/skeleton.EVGENtoHIT_ISF.py 2>&1 |tee log.EVNTtoHITS)

ls -ltrh $logfolder
for file in $(find $logfolder -type f -name '*.darshan')
do
    echo $file
    (cd $logfolder &&\
    $DARSHAN_BASE_DIR/bin/darshan-parser --show-incomplete --base --perf $file > $file.txt && \
    $DARSHAN_BASE_DIR/bin/darshan-job-summary.pl $file)
    mv -f $file* $workdir
done



# LD_PRELOAD="$DARSHAN_BASE_DIR/lib/libdarshan.so" 
# (cd FullG4_ttbar && rm -rf * && ATHENA_PROC_NUMBER=$nproc Sim_tf.py --postInclude "default:RecJobTransforms/UseFrontier.py" --preExec "EVNTtoHITS:simFlags.SimBarcodeOffset.set_Value_and_Lock(200000)" "EVNTtoHITS:simFlags.TRTRangeCut=30.0;simFlags.TightMuonStepping=True" --preInclude "EVNTtoHITS:SimulationJobOptions/preInclude.BeamPipeKill.py,SimulationJobOptions/preInclude.FrozenShowersFCalOnly.py" --physicsList=FTFP_BERT_ATL_VALIDATION --randomSeed=2357 --DBRelease="all:current" --conditionsTag "default:OFLCOND-MC16-SDR-14" --geometryVersion="default:ATLAS-R2-2016-01-00-01_VALIDATION" --runNumber=700403 --DataRunNumber=284500 --simulator=FullG4 --truthStrategy=MC15aPlus  --inputEVNTFile "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/SimCoreTests/valid1.410000.PowhegPythiaEvtGen_P2012_ttbar_hdamp172p5_nonallhad.evgen.EVNT.e4993.EVNT.08166201._000012.pool.root.1" --outputHITSFile "test.HITS.pool.root" --maxEvents ${nevents}  --imf False --asetup AtlasOffline,21.0.15 --athenaopts='--nprocs='${nproc}' --preloadlib=/lcrc/group/ATLAS/users/rwang/Argonne_computing/PPS-CCE/darshan/build_darshan/darshan-3.4.0-pre1/lib/libdarshan.so')