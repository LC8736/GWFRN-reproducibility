# Third-party software and rights

These references identify independent software/method sources, not manuscript
authors. Do not remove upstream copyright or licence notices for anonymisation.

| Component | Source/version | What is executed |
|---|---|---|
| PySAL mgwr | https://github.com/pysal/mgwr ; 2.2.1 | GWR Gaussian fitting and external-location prediction; MGWR is not run |
| GNNWR | https://github.com/zjuwss/gnnwr ; commit `2711bdee9821bfe242992d2eb885a78be1cf2628`; installed distribution 0.1.17 | simulation author GNNWR/SWNN classes with an OLS/I/O adapter; empirical code uses official SWNN definitions and a disclosed training adapter |
| FastSGWR | https://github.com/Lessani252/FastSGWR ; commit `965a0dadb7a9d1c18de323d62f7a41236286734a` | simulation native ALPHA selection class and held-out WLS adapter; empirical implementation uses its kernel-mixture equations with spatial validation |
| FuncNN | https://github.com/cran/FuncNN ; commit `1dfe17b5f8b3d92f568709b03675114108d1bd11` | simulation PyTorch port of `fnn.fit` defaults; source R function retained for inspection, not required at runtime |
| SANNWR | https://www.mdpi.com/2220-9964/11/12/620 | disclosed paper-based SAPDNN–SWNN reproduction, not an author-supplied software package |

Only necessary upstream source files are bundled under `simulation/upstream/`,
with their GNU GPL-3.0 or MIT notices as supplied. The FuncNN DESCRIPTION records
GPL-3; the accompanying GNU licence text is retained with its source excerpt.
Installed packages retain their respective licences. No virtual environment,
binary runtime, credentials, browser state or copied upstream Git history is
redistributed. Source commits and fixed dependency versions identify the
implementations used; their inclusion does not imply endorsement.

No new ownership or permissive relicensing of upstream code is claimed. The
assembled dataset's public-sharing permission does not authorise relicensing
third-party components. This review deposit grants no additional rights beyond
the applicable existing licences; a final public release should declare a
compatible licence for original repository code after the rights holder's
approval. An anonymous review copy is distinct from an archival, licensed public
software release.
