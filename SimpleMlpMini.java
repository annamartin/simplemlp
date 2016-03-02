/*
 * ------------------------------------------------------------------------
 *
 *  Copyright by
 *  University of Konstanz, Germany and
 *  KNIME GmbH, Konstanz, Germany
 *  Website: http://www.knime.org; Email: contact@knime.org
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License, Version 3, as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses>.
 *
 *  Additional permission under GNU GPL version 3 section 7:
 *
 *  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
 *  Hence, KNIME and ECLIPSE are both independent programs and are not
 *  derived from each other. Should, however, the interpretation of the
 *  GNU GPL Version 3 ("License") under any applicable laws result in
 *  KNIME and ECLIPSE being a combined program, KNIME GMBH herewith grants
 *  you the additional permission to use and propagate KNIME together with
 *  ECLIPSE with only the license terms in place for ECLIPSE applying to
 *  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
 *  license terms of ECLIPSE themselves allow for the respective use and
 *  propagation of ECLIPSE together with KNIME.
 *
 *  Additional permission relating to nodes for KNIME that extend the Node
 *  Extension (and in particular that are based on subclasses of NodeModel,
 *  NodeDialog, and NodeView) and that only interoperate with KNIME through
 *  standard APIs ("Nodes"):
 *  Nodes are deemed to be separate and independent programs and to not be
 *  covered works.  Notwithstanding anything to the contrary in the
 *  License, the License does not apply to Nodes, you are not required to
 *  license Nodes under the License, and you are granted a license to
 *  prepare and propagate Nodes, in each case even if such Nodes are
 *  propagated with or for interoperation with KNIME.  The owner of a Node
 *  may freely choose the license terms applicable to such Node, including
 *  when such Node is propagated with or for interoperation with KNIME.
 * ---------------------------------------------------------------------
 *
 * History
 *   Nov 6, 2015 (annamartin): created
 */
package de.unikn.knime.stud.martin.simplemlp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 *
 * @author annamartin
 */
public class SimpleMlpMini implements SimpleMlp {

    private int m_nInput, m_nHidden, m_nOutput; //number of input, hidden and output neurons

    private Double[] m_input, m_hidden, m_output; //values of the neurons

    private double[][] m_weightsL1, m_weightsL2, m_deltaL1, m_deltaL2; //weights on L1 (input to hidden) and L2 (hidden to output) layers

    private double m_learningRate = 0.0001; //learning rate

    private double m_batchSize = 10; // mini-batch fraction

    private int m_numOfIterations;

    private Boolean m_shuffle;

    private Boolean m_lrdecay;

    private int m_lrdecayrate;

    /**
     * @param nInput
     * @param nHidden
     * @param nOutput
     * @param random
     * @param learnrate
     * @param batchSize
     * @param numOfIteration
     * @param shuffle
     * @param lrdecayrate
     * @param lrdecay
     */
    public SimpleMlpMini(final int nInput, final int nHidden, final int nOutput, final Random random,
                         final double learnrate, final double batchSize, final int numOfIteration,
                         final Boolean shuffle, final int lrdecayrate, final boolean lrdecay) {
        m_batchSize = batchSize;
        m_learningRate = learnrate;
        m_numOfIterations = numOfIteration;
        m_nInput = nInput;
        m_nHidden = nHidden;
        m_nOutput = nOutput;
        m_input = new Double[m_nInput + 1]; // + 1 for bias
        m_hidden = new Double[m_nHidden + 1];
        m_output = new Double[m_nOutput];
        m_weightsL1 = new double[m_nHidden + 1][m_nInput + 1];
        m_weightsL2 = new double[m_nOutput][m_nHidden + 1];
        m_deltaL1 = new double[m_nHidden + 1][m_nInput + 1];
        m_deltaL2 = new double[m_nOutput][m_nHidden + 1];
        m_shuffle = shuffle;
        m_lrdecay = lrdecay;
        m_lrdecayrate = lrdecayrate;

        generateWeights(random);
    }

    private void generateWeights(final Random random) {
        for (int i = 1; i <= m_nHidden; i++) {
            for (int j = 0; j <= m_nInput; j++) {
                m_weightsL1[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < m_nOutput; i++) {
            for (int j = 0; j <= m_nHidden; j++) {
                m_weightsL2[i][j] = random.nextDouble() - 0.5;
            }

        }
    }

    /**
     * @param samplesarr
     * @return
     */
    @Override
    public Double[] passExample(final Double[] samplesarr) {
        for (int i = 0; i < m_nInput; i++) {
            m_input[i + 1] = samplesarr[i];
        }
        m_input[0] = 1.0; //set bias
        m_hidden[0] = 1.0;
        //for each neuron on the hidden layer
        for (int j = 1; j <= m_nHidden; j++) {
            double weightedSum = 0.0; // set initial value of weighted sum to 0
            for (int i = 0; i <= m_nInput; i++) {
                //weighted sum for the neuron
                weightedSum += m_weightsL1[j][i] * m_input[i];
            }
            double a = Math.exp(weightedSum);
            double b = Math.exp(-weightedSum);
            m_hidden[j] = ((a - b) / (a + b));
        }
        //for each neuron on the output layer

        for (int j = 0; j < m_nOutput; j++) {
            double weightedSum = 0.0; // set initial value of weighted sum to 0
            for (int i = 0; i <= m_nHidden; i++) {
                //weighted sum for the neuron
                weightedSum += m_weightsL2[j][i] * m_hidden[i];
            }
            double a = Math.exp(weightedSum);
            double b = Math.exp(-weightedSum);
            m_output[j] = ((a - b) / (a + b));
        }
        return m_output;

    }

    /**
     * @param outputsarr
     */
    private void backPropagate(final Double[] outputsarr) {
        double[] errorL2 = new double[m_nOutput + 1];
        double[] errorL1 = new double[m_nHidden + 1];
        double errorSum = 0.0;

        for (int i = 0; i < m_nOutput; i++) {
            errorL2[i] = (1 - (m_output[i] * m_output[i])) * (outputsarr[i] - m_output[i]);
        }

        for (int i = 1; i <= m_nHidden; i++) {
            for (int j = 0; j < m_nOutput; j++) {
                // !!!
                errorSum += m_weightsL2[j][i] * errorL2[j];
            }
            errorL1[i] = (1 - m_hidden[i] * m_hidden[i]) * errorSum;
            errorSum = 0.0;
        }

        //update deltas
        for (int i = 0; i < m_nOutput; i++) {
            for (int j = 0; j <= m_nHidden; j++) {
                m_deltaL2[i][j] += errorL2[i] * m_hidden[j];
            }
        }
        for (int i = 1; i <= m_nHidden; i++) {
            for (int j = 0; j <= m_nInput; j++) {
                m_deltaL1[i][j] += errorL1[i] * m_input[j];
            }
        }
    }

    /**
     * @param samplesarr
     * @param outputsarr
     */
    @Override
    public void trainNetwork(final Double[][] samplesarr, final Double[][] outputsarr, final int iteration) {

        List<Integer> indexArray = new ArrayList<Integer>();

        if (m_shuffle) {
            // shuffle examples
            Integer[] array = new Integer[samplesarr.length];
            for (int i = 0; i < samplesarr.length; i++) {
                array[i] = i;
            }
            indexArray = Arrays.asList(array);
            Collections.shuffle(indexArray);
        }
        m_batchSize = Math.min(samplesarr.length, m_batchSize);
        int numBatches = (int)(samplesarr.length / m_batchSize);

        for (int i = 0; i < numBatches; i++) {

            //set delta arrays to 0 for each batch
            setDeltasToZero();

            // calculate outputs and delta arrays for each batch
            for (int j = (int)(i * m_batchSize); j < (i + 1) * m_batchSize; j++) {
                if (m_shuffle) {
                    passExample(samplesarr[indexArray.get(j)]);
                    backPropagate(outputsarr[indexArray.get(j)]);
                } else {
                    passExample(samplesarr[j]);
                    backPropagate(outputsarr[j]);
                }
            }

            // finally update weights on the 2nd and 1st layers
            updateWeights(iteration);

        }

        //handle reminder
        int reminder = (int)(samplesarr.length % m_batchSize);
        if (reminder > 0) {
            setDeltasToZero();
            for (int r = (int)(numBatches * m_batchSize - 1); r < samplesarr.length; r++) {
                if (m_shuffle) {
                    passExample(samplesarr[indexArray.get(r)]);
                    backPropagate(outputsarr[indexArray.get(r)]);
                } else {
                    passExample(samplesarr[r]);
                    backPropagate(outputsarr[r]);
                }
            }
            updateWeights(iteration);
        }

    }

    /**
     *
     */
    public void setDeltasToZero() {
        for (int m = 0; m < m_deltaL1.length; m++) {
            for (int n = 0; n < m_deltaL1[0].length; n++) {
                m_deltaL1[m][n] = 0.0;
            }
        }

        for (int m = 0; m < m_deltaL2.length; m++) {
            for (int n = 0; n < m_deltaL2[0].length; n++) {
                m_deltaL2[m][n] = 0.0;
            }
        }
    }

    /**
     * @param iteration
     *
     */
    public void updateWeights(final int iteration) {

        int d;
        if (m_lrdecay) {
            d = m_lrdecayrate;
        } else {
            d = 0;
        }

        for (int k = 0; k < m_weightsL2.length; k++) {
            for (int l = 0; l < m_weightsL2[0].length; l++) {
                m_weightsL2[k][l] = m_weightsL2[k][l] + m_deltaL2[k][l] * (m_learningRate / (1 + iteration * d));
            }
        }

        for (int k = 0; k < m_weightsL1.length; k++) {
            for (int l = 0; l < m_weightsL1[0].length; l++) {
                m_weightsL1[k][l] = m_weightsL1[k][l] + m_deltaL1[k][l] * (m_learningRate / (1 + iteration * d));
            }
        }
    }

    /**
     * @param input
     * @return
     */
    @Override
    public Double[] predict(final Double[] input) {
        return passExample(input);
    }

}
