using MGroup.AISolve.Core;
using MGroup.AISolve.MachineLearning;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.PreconditionedConjugateGradient;
using MGroup.LinearAlgebra.Iterative.Preconditioning;
using MGroup.LinearAlgebra.Iterative.Termination.Iterations;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MachineLearning.TensorFlow;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MGroup.AISolve.LinearAlgebra
{
	/// <summary>
	/// Implements the <see href="https://arxiv.org/abs/2207.02543">POD2G algorithm</see> for the solution of 
	/// parameterized linear steady state computational mechanics problems.
	/// </summary>
	public class Pod2GLinearSystemResponse : Pod2GResponse
	{
		private bool useInitialSolutionFromSurrogate;
		private int count;

		/// <summary>
		/// The object implementing <see cref="ILinearSystemCreator"/> that creates the components of a linear system
		/// based on the parameters as obtained from <see cref="IModelResponse.GetModelResponse"/>.
		/// </summary>
		public IList<IterativeStatistics> IterativeStatistics { get; protected set; }

		/// <summary>
		/// The object implementing <see cref="ILinearSystemCreator"/> that creates the components of a linear system
		/// based on the parameters as obtained from <see cref="IModelResponse.GetModelResponse"/>.
		/// </summary>
		public ILinearSystemCreator ModelCreator { get; protected set; }

		/// <summary>
		/// Constructs a <see cref="Pod2GLinearSystemResponse"/> object that implements the
		/// <see href="https://arxiv.org/abs/2207.02543">POD2G algorithm</see> for the solution of 
		/// parameterized linear steady state computational mechanics problems.
		/// </summary>
		/// <param name="podPrincipalComponents">The number of components that the set of raw solutions will be decomposed when POD is applied.</param>
		/// <param name="modelCreator">
		/// The object implementing <see cref="ILinearSystemCreator"/> that creates a tuple of a <see cref="CsrMatrix"/> and <see cref="Vector"/>
		/// based on the parameters as obtained from <see cref="IModelResponse.GetModelResponse"/>.
		/// </param>
		public Pod2GLinearSystemResponse(int podPrincipalComponents, ILinearSystemCreator modelCreator, bool useInitialSolutionFromSurrogate = false) 
			: base(podPrincipalComponents)
		{
			this.useInitialSolutionFromSurrogate = useInitialSolutionFromSurrogate;
			this.ModelCreator = modelCreator;
			IterativeStatistics = new List<IterativeStatistics>();
		}

		public override double[] GetModelResponse(double[] parameterValues, bool useAmgPreconditioner)
		{
			if (ModelCreator == null)
			{
				throw new InvalidOperationException("ModelCreator is null");
			}

			var model = ModelCreator.GetModel(parameterValues);
			var pcgBuilder = new ReorthogonalizedPcg.Builder
			{
				ResidualTolerance = 1E-6,
				MaxIterationsProvider = new PercentageMaxIterationsProvider(maxIterationsOverMatrixOrder: 0.2d),
			};

			IPreconditioner M = null;
			Vector xComputed = null;
			if (useAmgPreconditioner)
			{
				M = amgPreconditionerFactory.CreatePreconditionerFor(model.Item1);
				if (useInitialSolutionFromSurrogate) 
				{
					double[] prediction = CaeFffnSurrogate.Predict(parameterValues);
					xComputed = Vector.CreateFromArray(prediction);
				}
				else
				{
					xComputed = Vector.CreateZero(model.Item1.NumRows);
				}
			}
			else
			{
				count++;
				M = new JacobiPreconditioner(model.Item1.GetDiagonalAsArray());
				xComputed = Vector.CreateZero(model.Item1.NumRows);
			}

			var pcg = pcgBuilder.Build();
			IterativeStatistics stats = new IterativeStatistics() { HasConverged = false };
			try
			{
				stats = pcg.Solve(model.Item1, M, model.Item2, xComputed, useAmgPreconditioner == false && useInitialSolutionFromSurrogate == false, 
					() => Vector.CreateZero(model.Item2.Length));
			}
			catch (Exception ex) 
			{
				stats.AlgorithmName += " - " + ex.Message;
			}

			IterativeStatistics.Add(stats);
			int numPcgIterations = stats.NumIterationsRequired;
			int solutionLength = model.Item2.Length;
			Debug.WriteLine($"Number of PCG iterations = {numPcgIterations}. Dofs = {solutionLength}.");

			return xComputed.RawData;
		}

		public override string ToString()
		{
			var stats = IterativeStatistics.Take(count).ToArray();
			var statsConverged = stats.Where(x => x.HasConverged).ToArray();
			var statsAI = IterativeStatistics.Skip(count).ToArray();
			var statsAIConverged = statsAI.Where(x => x.HasConverged).ToArray();
			int min = statsConverged.Min(x => x.NumIterationsRequired);
			int max = statsConverged.Min(x => x.NumIterationsRequired);
			double avg = statsConverged.Sum(x => x.NumIterationsRequired) / statsConverged.Length;
			int minAI = statsAIConverged.Min(x => x.NumIterationsRequired);
			int maxAI = statsAIConverged.Min(x => x.NumIterationsRequired);
			double avgAI = statsAIConverged.Sum(x => x.NumIterationsRequired) / statsAIConverged.Length;

			return $"Min {min}, max {max}, avg {avg}, minAI {minAI}, maxAI {maxAI}, avgAI {avgAI}, NC {stats.Length - statsConverged.Length}, NCAI {statsAI.Length - statsAIConverged.Length}";
		}
	}
}
