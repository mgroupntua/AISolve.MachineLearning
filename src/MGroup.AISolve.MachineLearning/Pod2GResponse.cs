using MGroup.AISolve.Core;
using MGroup.LinearAlgebra.Iterative.AlgebraicMultiGrid;
using MGroup.LinearAlgebra.Iterative.AlgebraicMultiGrid.PodAmg;
using MGroup.LinearAlgebra.Iterative.AlgebraicMultiGrid.Smoothing;
using MGroup.LinearAlgebra.Iterative.GaussSeidel;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MachineLearning.TensorFlow;
using MGroup.MachineLearning.Utilities;
using System;
using System.Collections.Generic;

namespace MGroup.AISolve.MachineLearning
{
	/// <summary>
	/// Implements the <see href="https://arxiv.org/abs/2207.02543">POD2G algorithm</see> for the solution of 
	/// parameterized linear steady state computational mechanics problems.
	/// </summary>
	public abstract class Pod2GResponse : IModelResponse, IAIResponse
	{
		protected IList<double[]> trainingModelParameters = new List<double[]>();
		protected IList<Vector> trainingSolutionVectors = new List<Vector>();
		protected PodAmgPreconditioner.Factory amgPreconditionerFactory;

		/// <summary>
		/// The number of components that the set of raw solutions will be decomposed when POD is applied.
		/// </summary>
		public int PodPrincipalComponents { get; }

		/// <summary>
		/// The <see cref="CaeFffnSurrogate"/> that will be used by the <see cref="PodAmgPreconditioner"/> of the POD2G algorithn.
		/// </summary>
		public CaeFffnSurrogate CaeFffnSurrogate { get; }

		/// <summary>
		/// Constructs a <see cref="Pod2GResponse"/> object that implements the
		/// <see href="https://arxiv.org/abs/2207.02543">POD2G algorithm</see> for the solution of 
		/// parameterized linear steady state computational mechanics problems.
		/// </summary>
		/// <param name="podPrincipalComponents">The number of components that the set of raw solutions will be decomposed when POD is applied.</param>
		public Pod2GResponse(int podPrincipalComponents)
		{
			var datasetSplitter = new DatasetSplitter()
			{
				MinTestSetPercentage = 0.2,
				MinValidationSetPercentage = 0,
			};
			datasetSplitter.SetOrderToContiguous(DataSubsetType.Training, DataSubsetType.Test);

			var surrogateBuilder = new CaeFffnSurrogate.Builder()
			{
				CaeBatchSize = 20,
				CaeLearningRate = 0.001f,
				CaeNumEpochs = 50, //paper used 500
				DecoderFiltersWithoutOutput = new int[] { 64, 128, 256 },
				EncoderFilters = new int[] { 256, 128, 64, 32 },
				FfnnBatchSize = 20,
				FfnnHiddenLayerSize = 64,
				FfnnLearningRate = 0.0001f,
				FfnnNumEpochs = 300, // paper used 3000
				FfnnNumHiddenLayers = 6,
				Splitter = datasetSplitter,
			};

			this.PodPrincipalComponents = podPrincipalComponents;
			this.CaeFffnSurrogate = surrogateBuilder.BuildSurrogate();
			this.amgPreconditionerFactory = new PodAmgPreconditioner.Factory()
			{
				NumIterations = 1,
				SmootherBuilder = new GaussSeidelSmoother.Builder(new GaussSeidelIterationCsrSerial.Builder(), GaussSeidelSweepDirection.Symmetric, numIterations: 1),
				KeepOnlyNonZeroPrincipalComponents = true,
			};
		}

		/// <inheritdoc/>
		double[] IModelResponse.GetModelResponse(double[] parameterValues) => GetModelResponse(parameterValues, false);

		/// <inheritdoc/>
		double[] IAIResponse.GetModelResponse(double[] parameterValues) => GetModelResponse(parameterValues, true);

		public abstract double[] GetModelResponse(double[] parameterValues, bool isAIEnhanced);

		/// <summary>
		/// Stores the <paramref name="response"/> of a model having the specific <paramref name="parameterValues"/>.
		/// This information will be used when <see cref="TrainWithRegisteredModelResponses"/> is called by <see cref="AISolver"/>.
		/// </summary>
		/// <param name="parameterValues">A double array containing arbitrary parameters ordered as in <see cref="IModelResponse.GetModelResponse"/>.</param>
		/// <param name="response">A double array with the raw values of the solution of the linear system constructed by <see cref="InitializeProblem"/>.</param>
		public void RegisterModelResponse(double[] parameterValues, double[] response)
        {
            trainingModelParameters.Add(parameterValues);
            trainingSolutionVectors.Add(Vector.CreateFromArray(response));
        }

        /// <summary>
        /// Performs POD on the solution vectors and trains a <see cref="CaeFffnSurrogate"/> using the data
        /// obtained from <see cref="RegisterModelResponse"/>.
        /// </summary>
        public void TrainWithRegisteredModelResponses()
        {
            // Gather all previous solution vectors as columns of a matrix
            int numSamples = trainingSolutionVectors.Count;
            int numDofs = trainingSolutionVectors[0].Length;
            Matrix solutionVectors = Matrix.CreateZero(numDofs, numSamples);
            for (int j = 0; j < numSamples; ++j)
            {
                solutionVectors.SetSubcolumn(j, trainingSolutionVectors[j]);
            }

            // Free up some memory by deleting the stored solution vectors
            trainingSolutionVectors.Clear();

            // AMG-POD training
            amgPreconditionerFactory.Initialize(solutionVectors, PodPrincipalComponents);

            // Gather all previous model parameters
            if (trainingModelParameters.Count != numSamples)
            {
                throw new Exception($"Have gathered {trainingModelParameters.Count} sets of model parameters, " +
                    $"but {numSamples} solution vectors, while using initial preconditioner.");
            }

            int numParameters = trainingModelParameters[0].Length;
            var parametersAsArray = new double[numSamples, numParameters];
            for (int i = 0; i < numSamples; ++i)
            {
                if (trainingModelParameters[i].Length != numParameters)
                {
                    throw new Exception("The model parameter sets do not all have the same size");
                }

                for (int j = 0; j < numParameters; ++j)
                {
                    parametersAsArray[i, j] = trainingModelParameters[i][j];
                }
            }

            // CAE-FFNN training. Dimension 0 must be the number of samples.
            double[,] solutionsAsArray = solutionVectors.Transpose().CopytoArray2D();
            CaeFffnSurrogate.TrainAndEvaluate(parametersAsArray, solutionsAsArray, null);
        }
	}
}
