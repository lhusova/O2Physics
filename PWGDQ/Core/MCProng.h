// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
// Class to store the options for a Monte-Carlo prong history.
// First particle (index 0) in the std::vector corresponds to the most recent particle
// and continues with mother, grand-mother, ...
// Data members are: PDG codes, whether to check both charges, whether to exclude the specified PDG code,
//          bit maps with a bit dedicated to each source (MCProng::Source), bit map on whether the specified source to be excluded,
//          whether to use AND among all specified source requirements

/* The PDG codes us the PYTHIA standard.
A few non-existent PYTHIA codes are used to select more than one PYTHIA code.

0 - default, accepts all PYTHIA codes
100 - light unflavoured mesons in the code range 100-199
101 - all light and strange mesons in the code range 100-399
200 -        --"--                               200-299
300 - strange mesons in the code range           300-399
400 - charmed mesons in the code range           400-499
401 - open charm mesons (all D and D* mesons)    400-439
402 - open charm mesons and baryons together     400-439, 4000-4399
403 - all (open- or hidden-) charm hadrons (mesons and baryons) in the range  400-499, 4000-4999    (! no psi' here)
404 - charged open charmed mesons w/o s-quark    410-419
405 - neutral open charmed mesons                420-429
406 - charged open charmed mesons with s-quark   430-439
500 - beauty mesons in the code range            500-599
501 - open beauty mesons                         500-549
502 - open beauty mesons and baryons             500-549, 5000-5499
503 - all beauty hadrons                         500-599, 5000-5999
504 - neutral open beauty mesons w/o s-quark     510-519
505 - charged open beauty mesons                 520-529
506 - charged open beauty mesons with s-quark    530-539
900 - LF mesons + Quarkonia for LMEE             111, 221, 331, 113, 223, 333, 443, 100443
901 - LF mesons             for LMEE             111, 221, 331, 113, 223, 333
902 - all open charm open beauty mesons+baryons  400-439, 500-549, 4000-4399, 5000-5499
903 - all hadrons in the code range              100-599, 1000-5999
904 - chic0, chic1 and chic2                     445, 100441, 200443
1000 - light unflavoured baryons in the code range 1000-1999
2000 -        --"--                                2000-2999
3000 - strange baryons in the code range           3000-3999
4000 - charmed baryons in the code range           4000-4999
4001 - open charm baryons                          4000-4399
5000 - beauty baryons in the code range            5000-5999
5001 - open beauty baryons                         5000-5499
*/

#ifndef PWGDQ_CORE_MCPRONG_H_
#define PWGDQ_CORE_MCPRONG_H_

#include "TNamed.h"
#include "TString.h"

#include <vector>
#include <iostream>
#include <map>

class MCProng
{
 public:
  enum Source {
    // TODO: add more sources, see Run-2 code
    kNothing = -1,
    kPhysicalPrimary = 0, // Physical primary, ALICE definition
    kProducedInTransport, // Produced during transport through the detector (e.g. GEANT)
    kProducedByGenerator, // Produced by generator (if not, then produced by GEANT)
    kFromBackgroundEvent, // Produced in the underlying event
    kHEPMCFinalState,     // HEPMC code 11
    kIsPowhegDYMuon,      // POWHEG muons based on Pythia Status Code (=23) -> Drell-Yan signal
    kNSources
  };

  static std::map<TString, int> fgSourceNames;

  enum Constants {
    kPDGCodeNotAssigned = 0
  };

  MCProng();
  explicit MCProng(int n);
  MCProng(int n, int m);
  MCProng(int n, std::vector<int> pdgs, std::vector<bool> checkBothCharges, std::vector<bool> excludePDG,
          std::vector<uint64_t> sourceBits, std::vector<uint64_t> excludeSource, std::vector<bool> useANDonSourceBitMap,
          bool checkGenerationsInTime = false, std::vector<int> checkIfPDGInHistory = {}, std::vector<bool> excludePDGInHistory = {});
  MCProng(const MCProng& c) = default;
  virtual ~MCProng() = default;

  void SetPDGcode(int generation, int code, bool checkBothCharges = false, bool exclude = false);
  void SetSources(int generation, uint64_t bits, uint64_t exclude = 0, bool useANDonSourceBits = true);
  void SetSourceBit(int generation, int sourceBit, bool exclude = false);
  void SetUseANDonSourceBits(int generation, bool option = true);
  void SetSignalInTime(bool intime = false); // set variable to check generations in time or back in time (default)
  void Print() const;
  bool TestPDG(int i, int pdgCode) const;
  bool ComparePDG(int pdg, int prongPDG, bool checkBothCharges = false, bool exclude = false) const;

  int fNGenerations;
  std::vector<int> fPDGcodes;
  std::vector<bool> fCheckBothCharges;
  std::vector<bool> fExcludePDG;
  std::vector<uint64_t> fSourceBits;
  std::vector<uint64_t> fExcludeSource;
  std::vector<bool> fUseANDonSourceBitMap;
  bool fCheckGenerationsInTime;
  std::vector<int> fPDGInHistory;
  std::vector<bool> fExcludePDGInHistory;

  ClassDef(MCProng, 2);
};
#endif // PWGDQ_CORE_MCPRONG_H_
