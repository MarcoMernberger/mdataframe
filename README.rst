============
mdataframe 2
============


A convenience package to wrap pandas dataframes and machine learning.

Description
===========

A convenience package to wrap pandas dataframes, machine learning and statistics
as well as mbf jobs. 

Plan:
The basic assumption is that we work with pandas dataframes as a main data structure.
The dataframe will at every point contain exactly the data needed for analysis.

Features that we want:

- use an mdataframe like a pandas dataframe

- Chaining of methods
  e.g. df_input.impute().scale().cluster() should be viable syntax
  
- extended syntax to use functions that are not prebuild in pandas, but can be used
  like a pandas function

- every method should work on dataframes, e.g. in a jupyter notebook without
  the use of mbf packages

- we want a set of simple functions that can be used as input to pandas transform()
  or apply() function

- some steps require the usage of a full dataframe and will return a full dataframe.
  As they are used on the complete dataframe (e.g. TMM normalization), it cannot be
  reasonably used with transform/apply, as this only works on one series at a time.
  Still we need to wrap things in a was that we can simply use the extended function syntax.

- need to encapsulate additional parameters (e.g. columns to use)

  -> Solution 1: classes that take the parameters
        tmm = TMM(parameters)
        df.morph(tmm)

  -> Solution 2: wrapper functions
        tmm = get_morph_callable(parameters)
        df.morph(tmm)
        df.tmm()
  -> the class solution seems to be more readable, use this

- we need a simple way to "jobify" the analysis
  -> idea: chain your df as you would do directly
  -> jobify: encapsulate the chain with a jobify function that turns it into a
     series of FileGeneratingJobs() or a single job with all dependencies 


Class structure:

MDataFrame -> wrapper for pandas dataframe ... or ... I could just subclass DataFrame, but this would interfere with column lookup ... 

Transformer -> class that are callable and can be applied to the dataframe

Jobifyer -> class that takes care of encapsulating the dataframe operation into a job.

TransformerSome functions require additional informations -> parameter to Transformer


Note
====
