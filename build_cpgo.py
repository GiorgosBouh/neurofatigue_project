# build_cpgo.py
# Export CPGO to .ttl (UTF-8, no BOM) and .owl (RDF/XML)
# pip install rdflib

from rdflib import Graph
from pathlib import Path

TTL_TEXT = r"""
@prefix : <http://example.org/gait-ontology#> .
@prefix go: <http://example.org/gait-ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

<http://example.org/gait-ontology> a owl:Ontology ;
  dcterms:title "Cross-Population Gait Ontology (CPGO)"@en ;
  dcterms:description "Ontology for explaining gait impairments across populations (PD, ASD, Elderly, Stroke, CP), integrating biomechanical features and compensation strategies, designed to be used with ontology-constrained LLM reasoning."@en ;
  owl:versionInfo "0.2.0" ;
  owl:versionIRI <http://example.org/gait-ontology/0.2.0> ;
  dcterms:license <https://creativecommons.org/licenses/by/4.0/> ;
  dcterms:created "2025-09-24"^^xsd:date ;
  dcterms:modified "2025-09-24"^^xsd:date .

#################################################################
# CLASSES
#################################################################

:Entity a owl:Class ;
  rdfs:label "Entity"@en ;
  skos:definition "Top-level abstract superclass for all ontology elements." .

:GaitCase a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Gait Case"@en ;
  skos:definition "An abstract case/observation instance used to attach population, features, impairments, and strategies." .

:Population a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Population"@en ;
  skos:definition "A categorical grouping of individuals relevant to gait (e.g., PD, ASD, Elderly, Stroke, CP)." .

:GaitImpairment a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Gait Impairment"@en ;
  skos:definition "Clinically meaningful gait-level impairments." .

:BiomechanicalFeature a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Biomechanical Feature"@en ;
  skos:definition "A measurable/observable biomechanical parameter or pattern (spatiotemporal, kinematic, kinetic, EMG)." .

:CompensationStrategy a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Compensation Strategy"@en ;
  skos:definition "Observed or recommended strategy to mitigate an impairment (e.g., cueing, orthoses, gait modifications)." .

# Feature taxonomy
:SpatiotemporalParameter a owl:Class ;
  rdfs:subClassOf :BiomechanicalFeature ;
  rdfs:label "Spatiotemporal Parameter"@en ;
  skos:definition "Temporal/linear measures of gait cycles and steps." .

:KinematicFeature a owl:Class ;
  rdfs:subClassOf :BiomechanicalFeature ;
  rdfs:label "Kinematic Feature"@en ;
  skos:definition "Angular displacements/ROM and their patterns." .

:KineticFeature a owl:Class ;
  rdfs:subClassOf :BiomechanicalFeature ;
  rdfs:label "Kinetic Feature"@en ;
  skos:definition "Forces, moments and powers during gait." .

:EMGFeature a owl:Class ;
  rdfs:subClassOf :BiomechanicalFeature ;
  rdfs:label "EMG Feature"@en ;
  skos:definition "Electromyographic features linked to gait." .

# Anatomy & context
:Joint a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Joint/Segment"@en ;
  skos:definition "Joint or body segment used for classifying features." .

:PlaneOfMotion a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Plane of Motion"@en ;
  skos:definition "Reference planes: sagittal, frontal, transverse." .

:GaitEvent a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Gait Event"@en ;
  skos:definition "Events such as initial contact and toe-off." .

:Laterality a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Laterality"@en ;
  skos:definition "Laterality qualifiers (left/right/bilateral)." .

:GaitPhase a owl:Class ;
  rdfs:subClassOf :Entity ;
  rdfs:label "Gait Phase"@en ;
  skos:definition "Phases such as stance and swing." .

#################################################################
# OBJECT PROPERTIES (+ inverses)
#################################################################

:hasPopulation a owl:ObjectProperty , owl:FunctionalProperty ;
  rdfs:domain :GaitCase ;
  rdfs:range :Population ;
  rdfs:label "has population"@en ;
  skos:definition "Assigns the population category for a case." .

:isPopulationOf a owl:ObjectProperty ;
  owl:inverseOf :hasPopulation ;
  rdfs:domain :Population ;
  rdfs:range :GaitCase ;
  rdfs:label "is population of" .

:exhibitsFeature a owl:ObjectProperty ;
  rdfs:domain :GaitCase ;
  rdfs:range :BiomechanicalFeature ;
  rdfs:label "exhibits feature"@en ;
  skos:definition "Declares an observed biomechanical feature in a case." .

:isFeatureOf a owl:ObjectProperty ;
  owl:inverseOf :exhibitsFeature ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range :GaitCase ;
  rdfs:label "is feature of" .

:hasImpairment a owl:ObjectProperty ;
  rdfs:domain :GaitCase ;
  rdfs:range :GaitImpairment ;
  rdfs:label "has impairment"@en ;
  skos:definition "Declares an inferred or asserted gait impairment." .

:isImpairmentOf a owl:ObjectProperty ;
  owl:inverseOf :hasImpairment ;
  rdfs:domain :GaitImpairment ;
  rdfs:range :GaitCase ;
  rdfs:label "is impairment of" .

:recommendedStrategy a owl:ObjectProperty ;
  rdfs:domain :GaitCase ;
  rdfs:range :CompensationStrategy ;
  rdfs:label "recommended strategy" .

:strategyFor a owl:ObjectProperty ;
  owl:inverseOf :recommendedStrategy ;
  rdfs:domain :CompensationStrategy ;
  rdfs:range :GaitCase ;
  rdfs:label "strategy for" .

:observedCompensation a owl:ObjectProperty ;
  rdfs:domain :GaitCase ;
  rdfs:range :CompensationStrategy ;
  rdfs:label "observed compensation" .

:associatedWithFeature a owl:ObjectProperty ;
  rdfs:domain :GaitImpairment ;
  rdfs:range :BiomechanicalFeature ;
  rdfs:label "associated with feature" .

:isAssociatedWithImpairment a owl:ObjectProperty ;
  owl:inverseOf :associatedWithFeature ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range :GaitImpairment ;
  rdfs:label "is associated with impairment" .

:managedBy a owl:ObjectProperty ;
  rdfs:domain :GaitImpairment ;
  rdfs:range :CompensationStrategy ;
  rdfs:label "managed by" .

:manages a owl:ObjectProperty ;
  owl:inverseOf :managedBy ;
  rdfs:domain :CompensationStrategy ;
  rdfs:range :GaitImpairment ;
  rdfs:label "manages impairment" .

:affectsJoint a owl:ObjectProperty ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range :Joint ;
  rdfs:label "affects joint" .

:measuredInPlane a owl:ObjectProperty ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range :PlaneOfMotion ;
  rdfs:label "measured in plane" .

:hasGaitEvent a owl:ObjectProperty ;
  rdfs:domain :GaitCase ;
  rdfs:range :GaitEvent ;
  rdfs:label "has gait event" .

:hasLaterality a owl:ObjectProperty ;
  rdfs:domain :GaitCase ;
  rdfs:range :Laterality ;
  rdfs:label "has laterality" .

:hasPhase a owl:ObjectProperty ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range :GaitPhase ;
  rdfs:label "has phase" .

:typicalInPopulation a owl:ObjectProperty ;
  rdfs:domain :GaitImpairment ;
  rdfs:range :Population ;
  rdfs:label "typical in population" .

#################################################################
# DATA PROPERTIES
#################################################################

:hasValue a owl:DatatypeProperty ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range xsd:double ;
  rdfs:label "has value" .

:hasUnit a owl:DatatypeProperty ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range xsd:string ;
  rdfs:label "has unit" .

:thresholdLow a owl:DatatypeProperty ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range xsd:double ;
  rdfs:label "threshold low" .

:thresholdHigh a owl:DatatypeProperty ;
  rdfs:domain :BiomechanicalFeature ;
  rdfs:range xsd:double ;
  rdfs:label "threshold high" .

:definition a owl:DatatypeProperty ;
  rdfs:domain :Entity ;
  rdfs:range xsd:string ;
  rdfs:label "definition" .

#################################################################
# DISJOINTNESS (quality constraints)
#################################################################

[ a owl:AllDisjointClasses ;
  owl:members ( :Population :GaitImpairment :BiomechanicalFeature :CompensationStrategy :Joint :PlaneOfMotion :GaitEvent :Laterality :GaitPhase )
] .

#################################################################
# VALUE PARTITIONS / CONTROLLED VOCABS
#################################################################

# Joints
:Ankle a owl:Class ; rdfs:subClassOf :Joint ; rdfs:label "Ankle" .
:Knee a owl:Class ; rdfs:subClassOf :Joint ; rdfs:label "Knee" .
:Hip a owl:Class ; rdfs:subClassOf :Joint ; rdfs:label "Hip" .
:Pelvis a owl:Class ; rdfs:subClassOf :Joint ; rdfs:label "Pelvis" .
:Trunk a owl:Class ; rdfs:subClassOf :Joint ; rdfs:label "Trunk" .

# Planes
:Sagittal a owl:Class ; rdfs:subClassOf :PlaneOfMotion ; rdfs:label "Sagittal" .
:Frontal a owl:Class ; rdfs:subClassOf :PlaneOfMotion ; rdfs:label "Frontal" .
:Transverse a owl:Class ; rdfs:subClassOf :PlaneOfMotion ; rdfs:label "Transverse" .

# Phases & laterality
:Stance a owl:Class ; rdfs:subClassOf :GaitPhase ; rdfs:label "Stance" .
:Swing a owl:Class ; rdfs:subClassOf :GaitPhase ; rdfs:label "Swing" .
:Left a owl:Class ; rdfs:subClassOf :Laterality ; rdfs:label "Left" .
:Right a owl:Class ; rdfs:subClassOf :Laterality ; rdfs:label "Right" .
:Bilateral a owl:Class ; rdfs:subClassOf :Laterality ; rdfs:label "Bilateral" .

#################################################################
# POPULATIONS
#################################################################

:PD a owl:Class ;
  rdfs:subClassOf :Population ;
  rdfs:label "Parkinson's disease"@en ;
  skos:altLabel "PD" .

:ASD a owl:Class ;
  rdfs:subClassOf :Population ;
  rdfs:label "Autism Spectrum Disorder"@en ;
  skos:altLabel "ASD" .

:Elderly a owl:Class ;
  rdfs:subClassOf :Population ;
  rdfs:label "Elderly" .

:Stroke a owl:Class ;
  rdfs:subClassOf :Population ;
  rdfs:label "Stroke (Hemiparetic)"@en ;
  skos:altLabel "Post-stroke" .

:CP a owl:Class ;
  rdfs:subClassOf :Population ;
  rdfs:label "Cerebral Palsy"@en ;
  skos:altLabel "CP" .

#################################################################
# IMPAIRMENTS
#################################################################

:FreezingOfGait a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :PD ] ;
  rdfs:label "Freezing of Gait" .

:HemipareticFootDrop a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :Stroke ] ;
  rdfs:label "Hemiparetic Foot Drop" .

:CrouchGait a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :CP ] ;
  rdfs:label "Crouch Gait" .

:SpasticGait a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :CP ] ;
  rdfs:label "Spastic Gait" .

:Asymmetry a owl:Class ;
  rdfs:subClassOf :GaitImpairment ;
  rdfs:label "Gait Asymmetry" .

:ReducedStrideLength a owl:Class ;
  rdfs:subClassOf :GaitImpairment ;
  rdfs:label "Reduced Stride Length (Impairment)" .

:IncreasedDoubleSupport a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :Elderly ] ;
  rdfs:label "Increased Double Support" .

:MotorIncoordination a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :ASD ] ;
  rdfs:label "Motor Incoordination" .

# New impairments for richer coverage
:ShufflingGait a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :PD ] ;
  rdfs:label "Shuffling Gait" .

:CautiousGait a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :Elderly ] ;
  rdfs:label "Cautious Gait" .

:CircumductionGait a owl:Class ;
  rdfs:subClassOf :GaitImpairment ,
    [ a owl:Restriction ; owl:onProperty :typicalInPopulation ; owl:someValuesFrom :Stroke ] ;
  rdfs:label "Circumduction Gait" .

#################################################################
# FEATURES
#################################################################

# Spatiotemporal
:StrideLength a owl:Class ;
  rdfs:subClassOf :SpatiotemporalParameter ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Stride Length"@en ;
  :definition "Linear distance covered in one stride." .

:StepWidth a owl:Class ;
  rdfs:subClassOf :SpatiotemporalParameter ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Frontal ] ;
  rdfs:label "Step Width"@en ;
  :definition "Mediolateral distance between successive foot placements." .

:StepWidthVariability a owl:Class ;
  rdfs:subClassOf :SpatiotemporalParameter ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Frontal ] ;
  rdfs:label "Step Width Variability" .

:GaitSpeed a owl:Class ;
  rdfs:subClassOf :SpatiotemporalParameter ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Gait Speed" .

:DoubleSupportTime a owl:Class ;
  rdfs:subClassOf :SpatiotemporalParameter ;
  rdfs:label "Double Support Time" .

:StrideTimeVariability a owl:Class ;
  rdfs:subClassOf :SpatiotemporalParameter ;
  rdfs:label "Stride Time Variability" .

:StanceTimeAsymmetry a owl:Class ;
  rdfs:subClassOf :SpatiotemporalParameter ;
  rdfs:label "Stance Time Asymmetry" .

# Kinematics
:AnkleDorsiflexionROM a owl:Class ;
  rdfs:subClassOf :KinematicFeature ,
    [ a owl:Restriction ; owl:onProperty :affectsJoint ; owl:someValuesFrom :Ankle ] ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Ankle Dorsiflexion ROM" .

:KneeFlexionROM a owl:Class ;
  rdfs:subClassOf :KinematicFeature ,
    [ a owl:Restriction ; owl:onProperty :affectsJoint ; owl:someValuesFrom :Knee ] ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Knee Flexion ROM" .

:HipExtensionROM a owl:Class ;
  rdfs:subClassOf :KinematicFeature ,
    [ a owl:Restriction ; owl:onProperty :affectsJoint ; owl:someValuesFrom :Hip ] ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Hip Extension ROM" .

:ExcessiveKneeFlexion a owl:Class ;
  rdfs:subClassOf :KinematicFeature ,
    [ a owl:Restriction ; owl:onProperty :affectsJoint ; owl:someValuesFrom :Knee ] ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Excessive Knee Flexion" .

:ReducedAnkleDorsiflexion a owl:Class ;
  rdfs:subClassOf :KinematicFeature ,
    [ a owl:Restriction ; owl:onProperty :affectsJoint ; owl:someValuesFrom :Ankle ] ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Reduced Ankle Dorsiflexion" .

# Kinetics
:vGRFFlattening a owl:Class ;
  rdfs:subClassOf :KineticFeature ;
  rdfs:label "Flattened Vertical GRF"@en ;
  :definition "Attenuated double-hump pattern of vertical ground reaction force over stance." .

:PushOffPower a owl:Class ;
  rdfs:subClassOf :KineticFeature ,
    [ a owl:Restriction ; owl:onProperty :affectsJoint ; owl:someValuesFrom :Ankle ] ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Push-off Power" .

:ReducedPushOffPower a owl:Class ;
  rdfs:subClassOf :KineticFeature ,
    [ a owl:Restriction ; owl:onProperty :affectsJoint ; owl:someValuesFrom :Ankle ] ,
    [ a owl:Restriction ; owl:onProperty :measuredInPlane ; owl:someValuesFrom :Sagittal ] ;
  rdfs:label "Reduced Push-off Power" .

#################################################################
# STRATEGIES
#################################################################

# Cueing
:ExternalCueing a owl:Class ;
  rdfs:subClassOf :CueingStrategy ;
  rdfs:label "External Cueing" .

:MetronomeCueing a owl:Class ;
  rdfs:subClassOf :CueingStrategy ;
  rdfs:label "Metronome Cueing" .

:VisualCueing a owl:Class ;
  rdfs:subClassOf :CueingStrategy ;
  rdfs:label "Visual Cueing" .

# Assistive devices
:AnkleFootOrthosis a owl:Class ;
  rdfs:subClassOf :AssistiveDevice ;
  rdfs:label "Ankle-Foot Orthosis"@en ;
  skos:altLabel "AFO" .

:CaneUse a owl:Class ;
  rdfs:subClassOf :AssistiveDevice ;
  rdfs:label "Cane Use" .

# Gait modifications
:WiderStepWidth a owl:Class ;
  rdfs:subClassOf :GaitModification ;
  rdfs:label "Wider Step Width" .

:HipCircumduction a owl:Class ;
  rdfs:subClassOf :GaitModification ;
  rdfs:label "Hip Circumduction" .

:ConsciousStepLengthIncrease a owl:Class ;
  rdfs:subClassOf :GaitModification ;
  rdfs:label "Conscious Step Length Increase" .

:RigidGaitPattern a owl:Class ;
  rdfs:subClassOf :GaitModification ;
  rdfs:label "Rigid Gait Pattern"@en ;
  :definition "Reduced flexibility/variability adopted to stabilize timing in some individuals." .

:ToeWalkingStrategy a owl:Class ;
  rdfs:subClassOf :GaitModification ;
  rdfs:label "Toe Walking Strategy" .

#################################################################
# TYPICAL LINKS (Impairments <-> Features/Strategies)
#################################################################

:FreezingOfGait rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :vGRFFlattening ] ,
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :StrideLength ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :ExternalCueing ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :MetronomeCueing ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :VisualCueing ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :ConsciousStepLengthIncrease ] .

:HemipareticFootDrop rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :ReducedAnkleDorsiflexion ] ,
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :StanceTimeAsymmetry ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :AnkleFootOrthosis ] .

:CrouchGait rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :ExcessiveKneeFlexion ] ,
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :HipExtensionROM ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :ToeWalkingStrategy ] .

:SpasticGait rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :KneeFlexionROM ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :AnkleFootOrthosis ] .

:IncreasedDoubleSupport rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :DoubleSupportTime ] ,
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :ReducedPushOffPower ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :WiderStepWidth ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :CaneUse ] .

:MotorIncoordination rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :StrideTimeVariability ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :RigidGaitPattern ] .

:ShufflingGait rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :ReducedStrideLength ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :ConsciousStepLengthIncrease ] .

:CautiousGait rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :DoubleSupportTime ] ,
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :GaitSpeed ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :WiderStepWidth ] .

:CircumductionGait rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :ReducedAnkleDorsiflexion ] ,
  [ a owl:Restriction ; owl:onProperty :associatedWithFeature ; owl:someValuesFrom :StanceTimeAsymmetry ] ,
  [ a owl:Restriction ; owl:onProperty :managedBy ; owl:someValuesFrom :AnkleFootOrthosis ] .

#################################################################
# GCIs (Rule-like axioms on GaitCase)
#################################################################

# Existing 5 GCIs kept
# PD + vGRF flattening -> FoG (+ ExternalCueing)
[ a owl:Class ;
  owl:intersectionOf ( :GaitCase
                        [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:someValuesFrom :PD ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :vGRFFlattening ] )
] rdfs:subClassOf
  [ a owl:Class ;
    owl:intersectionOf (
      [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:someValuesFrom :FreezingOfGait ]
      [ a owl:Restriction ; owl:onProperty :recommendedStrategy ; owl:someValuesFrom :ExternalCueing ]
    )
  ] .

# Stroke + Reduced Ankle Dorsiflexion -> HemipareticFootDrop (+ AFO; HipCircumduction observed)
[ a owl:Class ;
  owl:intersectionOf ( :GaitCase
                        [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:someValuesFrom :Stroke ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :ReducedAnkleDorsiflexion ] )
] rdfs:subClassOf
  [ a owl:Class ;
    owl:intersectionOf (
      [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:someValuesFrom :HemipareticFootDrop ]
      [ a owl:Restriction ; owl:onProperty :recommendedStrategy ; owl:someValuesFrom :AnkleFootOrthosis ]
      [ a owl:Restriction ; owl:onProperty :observedCompensation ; owl:someValuesFrom :HipCircumduction ]
    )
  ] .

# Elderly + Reduced Push-off Power -> Increased Double Support (+ wider step width observed)
[ a owl:Class ;
  owl:intersectionOf ( :GaitCase
                        [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:someValuesFrom :Elderly ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :ReducedPushOffPower ] )
] rdfs:subClassOf
  [ a owl:Class ;
    owl:intersectionOf (
      [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:someValuesFrom :IncreasedDoubleSupport ]
      [ a owl:Restriction ; owl:onProperty :observedCompensation ; owl:someValuesFrom :WiderStepWidth ]
    )
  ] .

# ASD + Stride Time Variability -> Motor Incoordination (+ RigidGaitPattern observed)
[ a owl:Class ;
  owl:intersectionOf ( :GaitCase
                        [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:someValuesFrom :ASD ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :StrideTimeVariability ] )
] rdfs:subClassOf
  [ a owl:Class ;
    owl:intersectionOf (
      [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:someValuesFrom :MotorIncoordination ]
      [ a owl:Restriction ; owl:onProperty :observedCompensation ; owl:someValuesFrom :RigidGaitPattern ]
    )
  ] .

# CP + Excessive Knee Flexion -> Crouch Gait (+ ToeWalkingStrategy observed)
[ a owl:Class ;
  owl:intersectionOf ( :GaitCase
                        [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:someValuesFrom :CP ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :ExcessiveKneeFlexion ] )
] rdfs:subClassOf
  [ a owl:Class ;
    owl:intersectionOf (
      [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:someValuesFrom :CrouchGait ]
      [ a owl:Restriction ; owl:onProperty :observedCompensation ; owl:someValuesFrom :ToeWalkingStrategy ]
    )
  ] .

# New GCIs (3 extra)

# PD + ReducedStrideLength -> ShufflingGait (+ ConsciousStepLengthIncrease recommended)
[ a owl:Class ;
  owl:intersectionOf ( :GaitCase
                        [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:someValuesFrom :PD ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :ReducedStrideLength ] )
] rdfs:subClassOf
  [ a owl:Class ;
    owl:intersectionOf (
      [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:someValuesFrom :ShufflingGait ]
      [ a owl:Restriction ; owl:onProperty :recommendedStrategy ; owl:someValuesFrom :ConsciousStepLengthIncrease ]
    )
  ] .

# Elderly + DoubleSupportTime + low GaitSpeed -> CautiousGait
[ a owl:Class ;
  owl:intersectionOf ( :GaitCase
                        [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:someValuesFrom :Elderly ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :DoubleSupportTime ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :GaitSpeed ] )
] rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:someValuesFrom :CautiousGait ] .

# Stroke + ReducedAnkleDorsiflexion + StanceTimeAsymmetry -> CircumductionGait
[ a owl:Class ;
  owl:intersectionOf ( :GaitCase
                        [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:someValuesFrom :Stroke ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :ReducedAnkleDorsiflexion ]
                        [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:someValuesFrom :StanceTimeAsymmetry ] )
] rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:someValuesFrom :CircumductionGait ] .

#################################################################
# CARDINALITY GUIDANCE
#################################################################

:GaitCase rdfs:subClassOf
  [ a owl:Restriction ; owl:onProperty :hasPopulation ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onClass :Population ] ,
  [ a owl:Restriction ; owl:onProperty :exhibitsFeature ; owl:minQualifiedCardinality "0"^^xsd:nonNegativeInteger ; owl:onClass :BiomechanicalFeature ] ,
  [ a owl:Restriction ; owl:onProperty :hasImpairment ; owl:minQualifiedCardinality "0"^^xsd:nonNegativeInteger ; owl:onClass :GaitImpairment ] ,
  [ a owl:Restriction ; owl:onProperty :recommendedStrategy ; owl:minQualifiedCardinality "0"^^xsd:nonNegativeInteger ; owl:onClass :CompensationStrategy ] ,
  [ a owl:Restriction ; owl:onProperty :observedCompensation ; owl:minQualifiedCardinality "0"^^xsd:nonNegativeInteger ; owl:onClass :CompensationStrategy ] .
"""

def main():
    # Βάση στον home του τρέχοντος χρήστη (Ubuntu -> /home/ubuntu, Mac -> /Users/user)
    base_dir = Path.home() / "Documents" / "ontology_and_biomechanics"
    base_dir.mkdir(parents=True, exist_ok=True)  # φτιάχνει αν δεν υπάρχει

    ttl_path = base_dir / "gait_cpgo_clinical_v0_4_1.ttl"
    owl_path = base_dir / "gait_cpgo_clinical_v0_4_1.owl"

    g = Graph()
    g.parse(data=TTL_TEXT, format="turtle")
    g.serialize(destination=str(ttl_path), format="turtle")
    g.serialize(destination=str(owl_path), format="xml")

    print("Written:")
    print(" -", ttl_path)
    print(" -", owl_path)

if __name__ == "__main__":
    main()