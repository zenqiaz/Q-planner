MIT License

Copyright (c) 2026 Ruiqi Zhang, Qianying Liu, and Yuki Kurashige

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Scope and third-party terms

This licence covers the source code in this repository only. It does not cover:

- **ORCA** — the quantum-chemistry program invoked by this software is licensed
  separately by its authors (FACCTs GmbH / MPI für Kohlenforschung) and is not
  redistributed here. Users must obtain their own ORCA licence.
- **Benchmark reference data** — the W4-11, G2RC, FH51, TAUT15, MOR41 and TMC151
  reference values and geometries used in the accompanying paper's evaluation
  originate from their respective publications' supporting information and remain
  under those publishers' terms. They are not redistributed here as primary data.
- **Corpus input files** — calculation input files mined from NOMAD retain the
  licences assigned by their original depositors.
- **Model weights** — any base model used for fine-tuning is subject to its own
  licence (e.g. the Llama Community Licence), which this licence does not alter.

## A note on institutional copyright

Resolved 2026-10-07. The supervising author confirmed that the authors may release this
code under a licence of their choosing. The repository's full commit history
shows a single code author, so no contribution here was authored by an employee
of another institution: the collaborator's contribution to the associated paper
was computational resources (CRediT "Resources"), not code. The earlier concern
on this file -- that NII sits under ROIS, whose regulations on copyrighted works
cover computer programs, and that an institution might therefore hold rights in
part of this code -- does not arise on those facts.

The copyright line above names all three authors of the associated paper, which
is this group's convention. Strictly, copyright in the code follows its
authorship; narrow the line to the code's author if you prefer it exact.

Third-party components remain under their own terms and are not relicensed by
this file: ORCA, benchmark reference data, NOMAD input files, and model weights
(e.g. the Llama Community Licence).
