#!/bin/bash

# MulT-EHR Project Cleanup Script
# This script purges all training checkpoints, processed graph objects, 
# dataset pickles, and benchmark reports to ensure a clean state for a new run.

echo "Cleaning previous output and artifacts..."

# Remove previous results
rm -rf checkpoints/*
rm -rf data/graphs/*
rm -rf data/dataset_objects/*
rm -rf data/mimic3_objects/*
rm -rf benchmark/*

# Recreate essential directory structure
mkdir -p checkpoints
mkdir -p data/graphs
mkdir -p data/dataset_objects
mkdir -p data/mimic3_objects
mkdir -p benchmark/plots

echo "Cleanup completed successfully. Essential directories re-initialized."
