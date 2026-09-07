# darshan_reops
HEP-CCE/SOP darshan reops analysis

Enviroment
 - Tested with `python3.12` + ROOT `6-36-04` + `pydarshan 3.5.0`

Run
 - Identify and extract the reops of a given file

   ```python plot_offset_histogram.py <darshan_record> --include_names <filename>```

 - Identify and extract the reops of a given file and map them to ROOT object

   1. dump the ROOT clusters or branchs/fields of a given TTree/RNTuple

      ```python dump_cluster_boundary.py <paths-to-inputRootFile> <EventTreeName> <max_workers> <per_branch=0/1> <output=dump_{surfix}.json/dump_per_branch_{surfix}.json>```

   2. map the offsets

      ```python <darshan_record> --include_names inputRootFile  --enable_mapping --tree_branch_file <dump_{surfix}.json/dump_per_branch_{surfix}.json>```

   Note: The mapping checks if `per_branch_` is part of the `tree_branch_file` name to decide how to map the records and create the plots.