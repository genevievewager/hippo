# Q12 confound notes (accumulate)

- Report 1 decoding used `include_in_decoder`, which excluded
  `hippocampal_formation_transition` (SUB-mapped) units. Report 2 lab decoding
  did not apply that mask (all NON-SOMA regions). Report 4 uses the region map
  on **all** units and never `include_in_decoder`.
- Arena: lab boundary polygon vs sim equal-area square (`aspect` mismatch).
- Behaviour fit not yet applied (awaiting approval of parameter list / targets).
