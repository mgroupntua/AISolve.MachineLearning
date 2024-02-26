using MGroup.AISolve.Core;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Xunit;

namespace MGroup.AISolve.LinearAlgebra.Tests
{
    public class MatrixTests
    {
        private const int numAnalysesTotal = 300;
        private const int numSolutionsForTraining = 50; // paper used 300
        private const int numPrincipalComponents = 8;
        private const int seed = 13;
		private const double mean = 0.1;
		private const double stdev = 0.1;

		[Fact]
        public static void OmniColiseumMatrixTest()
        {
            // Do NOT execute on Azure DevOps
            if (Environment.GetEnvironmentVariable("SYSTEM_DEFINITIONID") != null)
            { 
                return; 
            }

            var modelParameters = GenerateParameterValues(numAnalysesTotal);
			string filename = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.FullName + @"\MGroup.AISolve.LinearAlgebra.Tests\Data\bcsstk14.mtx";
			var modelCreator = new FileLinearSystemCreator(filename, null, 0d, 0d);
			var modelResponse = new Pod2GLinearSystemResponse(numPrincipalComponents, modelCreator);
            var solver = new AISolver(numSolutionsForTraining, modelParameters, modelResponse, modelResponse);
            var responses = new List<double[]>(solver);

			Debug.WriteLine(modelResponse.ToString());
        }

        private static IList<double[]> GenerateParameterValues(int numAnalysesTotal)
        {
			var rng = new Random(seed);
            double[] paramsE = GenerateSamplesNormal(numAnalysesTotal, mean, stdev, rng).ToArray();

            var samples = new double[numAnalysesTotal][];
            for (int i = 0; i < numAnalysesTotal; i++)
            {
                samples[i] = new[] { paramsE[i] };
			}

			return samples;
        }

        private static IEnumerable<double> GenerateSamplesNormal(int count, double mean, double stddev, Random rng)
        {
            double[] samples = new double[count];
            MathNet.Numerics.Distributions.Normal.Samples(rng, samples, mean, stddev);
            return samples;
        }

    }
}
