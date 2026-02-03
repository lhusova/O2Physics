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

/// \file alice3strangenessFinder.cxx
///
/// \brief finding of V0 and cascade candidates for ALICE 3
///
/// This task finds and build condidates for strange hadrons (K0s, Lambda, AntiLambda, Xi-, Xi+, Omega-, Omega+)
/// using the output of the on-the-fly tracker.
///
/// \author Lucia Anna Tarasovičová, Pavol Jozef Šafárik University (SK)
///

#include "ALICE3/Core/TrackUtilities.h"
#include "ALICE3/DataModel/OTFPIDTrk.h"
#include "ALICE3/DataModel/OTFRICH.h"
#include "ALICE3/DataModel/OTFStrangeness.h"
#include "ALICE3/DataModel/OTFTOF.h"
#include "ALICE3/DataModel/tracksAlice3.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "Common/DataModel/TrackSelectionTables.h"

#include "DCAFitter/DCAFitterN.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "ReconstructionDataFormats/Track.h"
#include <Field/MagneticField.h>
#include <Framework/AnalysisHelpers.h>
#include <Framework/Configurable.h>
#include <Framework/HistogramRegistry.h>
#include <Framework/O2DatabasePDGPlugin.h>

#include <TGenPhaseSpace.h>
#include <TGeoGlobalMagField.h>
#include <TLorentzVector.h>
#include <TPDGCode.h>
#include <TRandom3.h>

#include <fairlogger/Logger.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace o2;
// using namespace o2::analysis;
using namespace o2::framework;
using namespace o2::constants::physics;

using Alice3TracksWPid = soa::Join<aod::Tracks, aod::TracksCov, aod::McTrackLabels, aod::TracksDCA, aod::UpgradeTrkPids, aod::UpgradeTofs, aod::UpgradeRichs>;
using Alice3Tracks = soa::Join<aod::StoredTracks, aod::StoredTracksCov, aod::McTrackLabels, aod::TracksDCA, aod::TracksCovExtension, aod::TracksAlice3>;
using Alice3MCParticles = soa::Join<aod::McParticles, aod::MCParticlesExtraA3>;
struct Alice3strangenessFinder {
  SliceCache cache;

  HistogramRegistry histos{"histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  Produces<aod::V0CandidateIndices> v0CandidateIndices; // contains V0 candidate indices
  Produces<aod::V0CandidateCores> v0CandidateCores;     // contains V0 candidate core information

  Configurable<float> nSigmaTOF{"nSigmaTOF", 5.0f, "Nsigma for TOF PID (if enabled)"};
  Configurable<float> dcaXYconstant{"dcaXYconstant", -1.0f, "[0] in |DCAxy| > [0]+[1]/pT"};
  Configurable<float> dcaXYpTdep{"dcaXYpTdep", 0.0, "[1] in |DCAxy| > [0]+[1]/pT"};
  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.45f, 0.5f, 0.55f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f, 1.0f, 1.05f, 1.1f}, "pt axis for QA histograms"};

  // Vertexing
  Configurable<bool> propagateToPCA{"propagateToPCA", false, "create tracks version propagated to PCA"};
  Configurable<bool> useAbsDCA{"useAbsDCA", true, "Minimise abs. distance rather than chi2"};
  Configurable<bool> useWeightedFinalPCA{"useWeightedFinalPCA", false, "Recalculate vertex position using track covariances, effective only if useAbsDCA is true"};
  Configurable<double> maxR{"maxR", 150., "reject PCA's above this radius"};
  Configurable<double> maxDZIni{"maxDZIni", 5, "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> maxDXYIni{"maxDXYIni", 4, "reject (if>0) PCA candidate if tracks DXY exceeds threshold"};
  Configurable<double> maxVtxChi2{"maxVtxChi2", 2, "reject (if>0) vtx. chi2 above this value"};
  Configurable<double> minParamChange{"minParamChange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> minRelChi2Change{"minRelChi2Change", 0.9, "stop iterations is chi2/chi2old > this"};
  // Operation
  Configurable<float> magneticField{"magneticField", 20.0f, "Magnetic field (in kilogauss)"};
  Configurable<bool> mcSameMotherCheck{"mcSameMotherCheck", true, "check if tracks come from the same MC mother"};
  // propagation options
  Configurable<bool> usePropagator{"usePropagator", false, "use external propagator"};
  Configurable<bool> refitWithMatCorr{"refitWithMatCorr", false, "refit V0 applying material corrections"};
  Configurable<bool> useCollinearV0{"useCollinearV0", true, "use collinear approximation for V0 fitting"};

  o2::vertexing::DCAFitterN<2> fitter;
  o2::vertexing::DCAFitterN<3> fitter3;

  Service<o2::framework::O2DatabasePDG> pdgDB;

  // partitions for D mesons
  Partition<Alice3Tracks> positiveSecondaryTracks =
    aod::track::signed1Pt > 0.0f && nabs(aod::track::dcaXY) > dcaXYconstant + dcaXYpTdep* nabs(aod::track::signed1Pt);
  Partition<Alice3Tracks> negativeSecondaryTracks =
    aod::track::signed1Pt < 0.0f && nabs(aod::track::dcaXY) > dcaXYconstant + dcaXYpTdep* nabs(aod::track::signed1Pt);
  Partition<Alice3MCParticles> positiveMCParticles = aod::mcparticle_alice3::charge > 0.0f;
  Partition<Alice3MCParticles> negativeMCParticles = aod::mcparticle_alice3::charge < 0.0f;
  // Partition<Alice3TracksWPid> negativeSecondaryPions = nabs(aod::upgrade_tof::nSigmaPionInnerTOF) < nSigmaTOF && nabs(aod::upgrade_tof::nSigmaPionOuterTOF) < nSigmaTOF && aod::track::signed1Pt < 0.0f && nabs(aod::track::dcaXY) > dcaXYconstant + dcaXYpTdep* nabs(aod::track::signed1Pt);
  // Partition<Alice3TracksWPid> positiveSecondaryPions = nabs(aod::upgrade_tof::nSigmaPionInnerTOF) < nSigmaTOF && nabs(aod::upgrade_tof::nSigmaPionOuterTOF) < nSigmaTOF && aod::track::signed1Pt > 0.0f && nabs(aod::track::dcaXY) > dcaXYconstant + dcaXYpTdep* nabs(aod::track::signed1Pt);
  // Partition<Alice3TracksWPid> secondaryProtons = nabs(aod::upgrade_tof::nSigmaProtonInnerTOF) < nSigmaTOF && nabs(aod::upgrade_tof::nSigmaProtonOuterTOF) < nSigmaTOF && aod::track::signed1Pt > 0.0f && nabs(aod::track::dcaXY) > dcaXYconstant + dcaXYpTdep* nabs(aod::track::signed1Pt);
  // Partition<Alice3TracksWPid> secondaryAntiProtons = nabs(aod::upgrade_tof::nSigmaProtonInnerTOF) < nSigmaTOF && nabs(aod::upgrade_tof::nSigmaProtonOuterTOF) < nSigmaTOF && aod::track::signed1Pt < 0.0f && nabs(aod::track::dcaXY) > dcaXYconstant + dcaXYpTdep* nabs(aod::track::signed1Pt);

  struct {
    float dcaDau;
    std::array<float, 3> posSV;
    std::array<float, 3> pV0;
    std::array<float, 3> pPos; // positive track
    std::array<float, 3> pNeg; // negative track
    float cosPA;
    float dcaToPV;
  } v0cand;

  void init(InitContext&)
  {
    // Initialization code here
    fitter.setBz(magneticField);
    fitter.setUseAbsDCA(useAbsDCA);
    fitter.setPropagateToPCA(propagateToPCA);
    fitter.setMaxR(maxR);
    fitter.setMinParamChange(minParamChange);
    fitter.setMinRelChi2Change(minRelChi2Change);
    fitter.setMaxDZIni(maxDZIni);
    fitter.setMaxDXYIni(maxDXYIni);
    fitter.setMaxChi2(maxVtxChi2);
    fitter.setUsePropagator(usePropagator);
    fitter.setRefitWithMatCorr(refitWithMatCorr);
    fitter.setCollinear(useCollinearV0);
    fitter.setMatCorrType(o2::base::Propagator::MatCorrType::USEMatCorrNONE);

    histos.add("hFitterQA", "", kTH1D, {{10, 0, 10}}); // For QA reasons, counting found candidates at different stages
    histos.add("hPt", "", kTH1D, {axisPt});
    histos.add("hEventCounter", "", kTH1D, {{1, 0, 2}});                         // counting processed events
    histos.add("hV0Counter", "", kTH1D, {{4, 0, 4}});                            // For QA reasons, counting found V0, 0: K0s, 1: Lambda, 2:AntiLambda, 3: wrongly identified V0
    histos.add("hRadiusVsHistNeg", "", kTH2D, {{100, 0, 150}, {12, 0.5, 12.5}}); // radius vs hist for MC studies
    histos.add("hRadiusVsHistPos", "", kTH2D, {{100, 0, 150}, {12, 0.5, 12.5}}); // radius vs hist for MC studies
    histos.print();
  }
  /// function to check if tracks have the same mother in MC
  template <typename TTrackType>
  bool checkSameMother(TTrackType const& track1, TTrackType const& track2)
  {
    bool sameMother = false;
    if (!track1.has_mcParticle() || !track2.has_mcParticle())
      return sameMother;
    auto mcParticle1 = track1.template mcParticle_as<aod::McParticles>();
    auto mcParticle2 = track2.template mcParticle_as<aod::McParticles>();
    if (mcParticle2.globalIndex() == mcParticle1.globalIndex()) { // for the V0 daughters we store the mc label of the mother particle in the daughter tracks
      sameMother = true;
    }
    return sameMother;
  }

  template <typename TTrackType>
  bool buildDecayCandidateTwoBody(TTrackType const& posTrack, TTrackType const& negTrack, int pdgCode = -1)
  {
    o2::track::TrackParCov posTrackCov;
    o2::track::TrackParCov negTrackCov;
    if constexpr (requires { posTrack.vx(); }) {
      std::vector<double> v0DecayVertex;
      v0DecayVertex.push_back(posTrack.vx());
      v0DecayVertex.push_back(posTrack.vy());
      v0DecayVertex.push_back(posTrack.vz());
      TLorentzVector pos = {posTrack.px(), posTrack.py(), posTrack.pz(), posTrack.e()};
      TLorentzVector neg = {negTrack.px(), negTrack.py(), negTrack.pz(), negTrack.e()};
      // switch (pdgCode) {
      //     case kK0Short:
      //       o2::upgrade::convertTLorentzVectorToO2Track(kPiPlus, pos, v0DecayVertex, posTrackCov, pdgDB);
      //       o2::upgrade::convertTLorentzVectorToO2Track(kPiMinus, neg, v0DecayVertex, negTrackCov, pdgDB);
      //       break;
      //     case kLambda0:
      //       o2::upgrade::convertTLorentzVectorToO2Track(kProton, pos, v0DecayVertex, posTrackCov, pdgDB);
      //       o2::upgrade::convertTLorentzVectorToO2Track(kPiMinus, neg, v0DecayVertex, negTrackCov, pdgDB);
      //       break;
      //     case kLambda0Bar:
      //       o2::upgrade::convertTLorentzVectorToO2Track(kPiPlus, pos, v0DecayVertex, posTrackCov, pdgDB);
      //       o2::upgrade::convertTLorentzVectorToO2Track(kProtonBar, neg, v0DecayVertex, negTrackCov, pdgDB);
      //       break;
      //     default:
      o2::upgrade::convertTLorentzVectorToO2Track(1, pos, v0DecayVertex, posTrackCov);
      o2::upgrade::convertTLorentzVectorToO2Track(-1, neg, v0DecayVertex, negTrackCov);
      //     break;
      // }
    } else {
      posTrackCov = getTrackParCov(posTrack);
      negTrackCov = getTrackParCov(negTrack);
    }

    histos.fill(HIST("hPt"), negTrackCov.getPt());
    histos.fill(HIST("hPt"), posTrackCov.getPt());

    histos.fill(HIST("hFitterQA"), 0.5);
    //}-{}-{}-{}-{}-{}-{}-{}-{}-{}
    // Move close to minima
    int nCand = 0;
    try {
      nCand = fitter.process(posTrackCov, negTrackCov);
    } catch (...) {
      return false;
    }
    histos.fill(HIST("hFitterQA"), 1.5);
    if (nCand == 0) {
      LOG(info) << "0 candidates found by fitter";
      return false;
    }
    histos.fill(HIST("hFitterQA"), 2.5);
    //}-{}-{}-{}-{}-{}-{}-{}-{}-{}
    if (!fitter.isPropagateTracksToVertexDone() && !fitter.propagateTracksToVertex()) {
      LOG(info) << "RejProp failed";
      return false;
    }
    histos.fill(HIST("hFitterQA"), 3.5);
    posTrackCov = fitter.getTrack(0);
    negTrackCov = fitter.getTrack(1);
    std::array<float, 3> posP;
    std::array<float, 3> negP;
    posTrackCov.getPxPyPzGlo(posP);
    negTrackCov.getPxPyPzGlo(negP);
    v0cand.dcaDau = std::sqrt(fitter.getChi2AtPCACandidate());
    if constexpr (requires { posTrack.vx(); }) {
      v0cand.pPos[0] = posTrack.px();
      v0cand.pPos[1] = posTrack.py();
      v0cand.pPos[2] = posTrack.pz();
      v0cand.pNeg[0] = negTrack.px();
      v0cand.pNeg[1] = negTrack.py();
      v0cand.pNeg[2] = negTrack.pz();
    } else {
      v0cand.pPos[0] = posP[0];
      v0cand.pPos[1] = posP[1];
      v0cand.pPos[2] = posP[2];
      v0cand.pNeg[0] = negP[0];
      v0cand.pNeg[1] = negP[1];
      v0cand.pNeg[2] = negP[2];
    }
    v0cand.pV0[0] = v0cand.pPos[0] + v0cand.pNeg[0];
    v0cand.pV0[1] = v0cand.pPos[1] + v0cand.pNeg[1];
    v0cand.pV0[2] = v0cand.pPos[2] + v0cand.pNeg[2];
    const auto posSV = fitter.getPCACandidatePos();
    v0cand.posSV[0] = posSV[0];
    v0cand.posSV[1] = posSV[1];
    v0cand.posSV[2] = posSV[2];

    return true;
  }
  float calculateDCAStraightToPV(float X, float Y, float Z, float Px, float Py, float Pz, float pvX, float pvY, float pvZ)
  {
    return std::sqrt((std::pow((pvY - Y) * Pz - (pvZ - Z) * Py, 2) + std::pow((pvX - X) * Pz - (pvZ - Z) * Px, 2) + std::pow((pvX - X) * Py - (pvY - Y) * Px, 2)) / (Px * Px + Py * Py + Pz * Pz));
  }
  void processFindV0CandidateNoPid(aod::Collision const& collision, Alice3Tracks const&, aod::McParticles const&)
  {
    auto negativeSecondaryTracksGrouped = negativeSecondaryTracks->sliceByCached(aod::track::collisionId, collision.globalIndex(), cache);
    auto positiveSecondaryTracksGrouped = positiveSecondaryTracks->sliceByCached(aod::track::collisionId, collision.globalIndex(), cache);

    histos.fill(HIST("hEventCounter"), 1.0);
    // LOG(info) << "Processing collision index: " << collision.globalIndex() << " with " << positiveSecondaryTracksGrouped.size() << " positive and " << negativeSecondaryTracksGrouped.size() << " negative secondary tracks.";

    for (auto const& posTrack : positiveSecondaryTracksGrouped) {
      if (!posTrack.isReconstructed()) {
        continue; // no ghost tracks
      }

      for (auto const& negTrack : negativeSecondaryTracksGrouped) {
        if (!negTrack.isReconstructed()) {
          continue; // no ghost tracks
        }

        // auto mcParticle1 = posTrack.template mcParticle_as<aod::McParticles>();

        // if (mcSameMotherCheck && !checkSameMother(posTrack, negTrack))
        // continue;
        if (!buildDecayCandidateTwoBody(posTrack, negTrack, -1))
          continue;
        v0cand.cosPA = RecoDecay::cpa(std::array{collision.posX(), collision.posY(), collision.posZ()}, std::array{v0cand.posSV[0], v0cand.posSV[1], v0cand.posSV[2]}, std::array{v0cand.pV0[0], v0cand.pV0[1], v0cand.pV0[2]});
        v0cand.dcaToPV = calculateDCAStraightToPV(
          v0cand.posSV[0], v0cand.posSV[1], v0cand.posSV[2],
          v0cand.pV0[0], v0cand.pV0[1], v0cand.pV0[2],
          collision.posX(), collision.posY(), collision.posZ());
        v0CandidateIndices(collision.globalIndex(),
                           posTrack.globalIndex(),
                           negTrack.globalIndex(),
                           -1);
        v0CandidateCores(
          v0cand.posSV[0], v0cand.posSV[1], v0cand.posSV[2],
          v0cand.pPos[0], v0cand.pPos[1], v0cand.pPos[2],
          v0cand.pNeg[0], v0cand.pNeg[1], v0cand.pNeg[2],
          v0cand.dcaDau, posTrack.dcaXY(), negTrack.dcaXY(),
          v0cand.cosPA, v0cand.dcaToPV);
        // if (mcParticle1.pdgCode() == kK0Short) {
        //   histos.fill(HIST("hV0Counter"), 0.5);
        // } else if (mcParticle1.pdgCode() == kLambda0) {
        //   histos.fill(HIST("hV0Counter"), 1.5);
        // } else if (mcParticle1.pdgCode() == kLambda0Bar) {
        //   histos.fill(HIST("hV0Counter"), 2.5);
        // } else {
        //   histos.fill(HIST("hV0Counter"), 3.5);
        // }
      }
    }
  }
  void processMCTrueFromACTS(aod::McCollision const& collision, Alice3MCParticles const&)
  {
    // // Example of processing MC truth information from ACTS
    // for (auto const& mcCollision : mcCollisions) {
    //   // Process each MC collision
    //   // LOG(info) << "MC Collision ID: " << mcCollision.globalIndex();
    //   // You can access properties of mcCollision here

    //   // Example: Loop over MC particles associated with this collision
    //   auto particlesInCollision = mcParticles.sliceBy(aod::mcparticle::collisionId, mcCollision.globalIndex());
    //   for (auto const& mcParticle : particlesInCollision) {
    //     // Process each MC particle
    //     // LOG(info) << "  MC Particle ID: " << mcParticle.globalIndex() << ", PDG Code: " << mcParticle.pdgCode();
    //     // You can access properties of mcParticle here
    //   }
    // }

    auto negativeMCParticlesGrouped = negativeMCParticles->sliceByCached(aod::mcparticle::mcCollisionId, collision.globalIndex(), cache);
    auto positiveMCParticlesGrouped = positiveMCParticles->sliceByCached(aod::mcparticle::mcCollisionId, collision.globalIndex(), cache);
    LOG(info) << "Processing collision index: " << collision.globalIndex() << " with " << positiveMCParticlesGrouped.size() << " positive and " << negativeMCParticlesGrouped.size() << " negative MC particles.";

    float radiusPos = 0.0f;
    float radiusNeg = 0.0f;
    bool isK0s = false;
    bool isLambda = false;
    bool isAntiLambda = false;
    int v0PdgCode = 0;
    int iPosPart = 0;
    for (auto const& posParticle : positiveMCParticlesGrouped) {
      radiusPos = std::hypot(posParticle.vx(), posParticle.vy());
      histos.fill(HIST("hRadiusVsHistPos"), radiusPos, posParticle.nHits());
      int iNegPart = 0;
      LOG(info) << "Pos daughter pdg " << posParticle.pdgCode();
      for (auto const& negParticle : negativeMCParticlesGrouped) {
        if (negParticle.pdgCode() == kElectron)
          continue;
        radiusNeg = std::hypot(negParticle.vx(), negParticle.vy());
        if (iNegPart == 0)
          histos.fill(HIST("hRadiusVsHistNeg"), radiusNeg, negParticle.nHits());
        if (iPosPart == 0)
          LOG(info) << "Neg daughter pdg " << negParticle.pdgCode();
        if (radiusPos == radiusNeg) {
          isK0s = (posParticle.pdgCode() == kPiPlus && negParticle.pdgCode() == kPiMinus);
          if (isK0s)
            v0PdgCode = kK0Short;
          isLambda = (posParticle.pdgCode() == kProton && negParticle.pdgCode() == kPiMinus);
          if (isLambda)
            v0PdgCode = kLambda0;
          isAntiLambda = (posParticle.pdgCode() == kPiPlus && negParticle.pdgCode() == kProtonBar);
          if (isAntiLambda)
            v0PdgCode = kLambda0Bar;
          if (isK0s || isLambda || isAntiLambda) {
            LOG(info) << "Found V0 candidate: " << v0PdgCode;
            if (!buildDecayCandidateTwoBody(posParticle, negParticle, v0PdgCode))
              continue;
            v0cand.cosPA = RecoDecay::cpa(std::array{collision.posX(), collision.posY(), collision.posZ()}, std::array{v0cand.posSV[0], v0cand.posSV[1], v0cand.posSV[2]}, std::array{v0cand.pV0[0], v0cand.pV0[1], v0cand.pV0[2]});
            v0cand.dcaToPV = calculateDCAStraightToPV(
              v0cand.posSV[0], v0cand.posSV[1], v0cand.posSV[2],
              v0cand.pV0[0], v0cand.pV0[1], v0cand.pV0[2],
              collision.posX(), collision.posY(), collision.posZ());
            v0CandidateIndices(collision.globalIndex(),
                               posParticle.globalIndex(),
                               negParticle.globalIndex(),
                               0);
            v0CandidateCores(
              v0cand.posSV[0], v0cand.posSV[1], v0cand.posSV[2],
              v0cand.pPos[0], v0cand.pPos[1], v0cand.pPos[2],
              v0cand.pNeg[0], v0cand.pNeg[1], v0cand.pNeg[2],
              v0cand.dcaDau, 0, 0,
              v0cand.cosPA, v0cand.dcaToPV);
            if (isK0s) {
              histos.fill(HIST("hV0Counter"), 0.5);
            } else if (isLambda) {
              histos.fill(HIST("hV0Counter"), 1.5);
            } else if (isAntiLambda) {
              histos.fill(HIST("hV0Counter"), 2.5);
            } else {
              histos.fill(HIST("hV0Counter"), 3.5);
            }
          }
        }
        iNegPart++;
      }
      iPosPart++;
    }
  }
  //    void processFindV0CandidateWithPid(aod::Collision const& collision, aod::McParticles const& mcParticles, Alice3TracksWPid const&)
  //     {
  //         auto negativeSecondaryPionsGrouped = negativeSecondaryPions->sliceByCached(aod::track::collisionId, collision.globalIndex(), cache);
  //         auto positiveSecondaryPionsGrouped = positiveSecondaryPions->sliceByCached(aod::track::collisionId, collision.globalIndex(), cache);
  //         auto secondaryProtonsGrouped = secondaryProtons->sliceByCached(aod::track::collisionId, collision.globalIndex(), cache);
  //         auto secondaryAntiProtonsGrouped = secondaryAntiProtons->sliceByCached(aod::track::collisionId, collision.globalIndex(), cache);
  //     }
  PROCESS_SWITCH(Alice3strangenessFinder, processFindV0CandidateNoPid, "find V0 without PID", true);
  PROCESS_SWITCH(Alice3strangenessFinder, processMCTrueFromACTS, "process MC truth from ACTS", false);
  // PROCESS_SWITCH(alice3strangenessFinder, processFindV0CandidateWithPid, "find V0 with PID", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<Alice3strangenessFinder>(cfgc)};
}
