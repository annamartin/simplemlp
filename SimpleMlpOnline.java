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
public class SimpleMlpOnline implements SimpleMlp {

    private int m_nInput, m_nHidden, m_nOutput; //number of input, hidden and output neurons

    private Double[] m_input, m_hidden, m_output; //values of the neurons

    private double[][] m_weightsL1, m_weightsL2; //weights on L1 (input to hidden) and L2 (hidden to output) layers

    private double m_learningRate; //learning rate

    private Boolean m_shuffle;

    private Boolean m_lrdecay;

    private int m_lrdecayrate;

    /**
     * @param nInput
     * @param nHidden
     * @param nOutput
     * @param random
     * @param learnrate
     * @param shuffle
     * @param lrdecayrate
     * @param lrdecay
     */
    public SimpleMlpOnline(final int nInput, final int nHidden, final int nOutput, final Random random,
                           final double learnrate, final Boolean shuffle, final int lrdecayrate,
                           final boolean lrdecay) {
        m_learningRate = learnrate;
        m_nInput = nInput;
        m_nHidden = nHidden;
        m_nOutput = nOutput;
        m_input = new Double[m_nInput + 1]; // + 1 for bias
        m_hidden = new Double[m_nHidden + 1];
        m_output = new Double[m_nOutput];
        m_weightsL1 = new double[m_nHidden + 1][m_nInput + 1];
        m_weightsL2 = new double[m_nOutput][m_nHidden + 1];
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
    public void backPropagate(final Double[] outputsarr, final int iteration) {
        double[] errorL2 = new double[m_nOutput + 1];
        double[] errorL1 = new double[m_nHidden + 1];
        double errorSum = 0.0;

        for (int i = 0; i < m_nOutput; i++) {
            errorL2[i] = (1 - (m_output[i] * m_output[i])) * (outputsarr[i] - m_output[i]);
        }

        for (int i = 1; i <= m_nHidden; i++) {
            for (int j = 0; j < m_nOutput; j++) {
                errorSum += m_weightsL2[j][i] * errorL2[j];
            }
            errorL1[i] = (1 - m_hidden[i] * m_hidden[i]) * errorSum;
            errorSum = 0.0;
        }

        int d;
        if (m_lrdecay) {
            d = m_lrdecayrate;
        } else {
            d = 0;
        }

        //update weights
        for (int i = 0; i < m_nOutput; i++) {
            for (int j = 0; j <= m_nHidden; j++) {
                m_weightsL2[i][j] += errorL2[i] * m_hidden[j] * (m_learningRate / (1 + iteration * d));
            }
        }
        for (int i = 1; i <= m_nHidden; i++) {
            for (int j = 0; j <= m_nInput; j++) {
                m_weightsL1[i][j] += errorL1[i] * m_input[j] * (m_learningRate / (1 + iteration * d));
                ;
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

        for (int i = 0; i < samplesarr.length; i++) {
            if (m_shuffle) {
                passExample(samplesarr[indexArray.get(i)]);
                backPropagate(outputsarr[indexArray.get(i)], iteration);
            } else {
                passExample(samplesarr[i]);
                backPropagate(outputsarr[i], iteration);
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
