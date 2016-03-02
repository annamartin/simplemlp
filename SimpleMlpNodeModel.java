/*
 * ------------------------------------------------------------------------
 *  Copyright by KNIME GmbH, Konstanz, Germany
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
 * -------------------------------------------------------------------
 *
 * History
 *   27.10.2005 (cebron): created
 */
package de.unikn.knime.stud.martin.simplemlp;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnDomain;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.DoubleValue;
import org.knime.core.data.RowIterator;
import org.knime.core.data.container.ColumnRearranger;
import org.knime.core.data.container.SingleCellFactory;
import org.knime.core.data.def.DoubleCell;
import org.knime.core.data.def.StringCell;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelDoubleBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelInteger;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.port.PortType;

/**
 * RPropNodeModel trains a MultiLayerPerceptron with resilient backpropagation.
 *
 * @author
 */
public class SimpleMlpNodeModel extends NodeModel {

    /**
     * Available learning methods
     */
    public enum m_learnMods {
        ONLINE, BATCH, MINI_BATCH
    };

    public static final String LMODE = "mode";

    private final SettingsModelString m_lmode = new SettingsModelString(LMODE, null);

    /**
     * In port of the optional PMMLModel.
     */
    public static final int INMODEL = 1;

    /**
     * The maximum number of possible iterations.
     */
    public static final int MAXNRITERATIONS = 1000000;

    /**
     * The default number of iterations.
     */
    public static final int DEFAULTITERATIONS = 100;

    /**
     * The default number of iterations.
     */
    public static final int DEFAULTNEURONSPERLAYER = 10;

    /**
     * Key to store the number of maximum iterations.
     */
    public static final String MAXITER_KEY = "maxiter";

    /**
     * Key to store learning rate value.
     */
    public static final String LEARNINGRATE_KEY = "learnrate";

    /**
     * Key to store batch size.
     */
    public static final String BATCHSIZE_KEY = "batchsize";

    /**
     * Key to store whether missing values should be ignored.
     */
    public static final String IGNOREMV_KEY = "ignoremv";

    /**
     * Key to store whether to shuffle training examples.
     */
    public static final String SHUFFLE_KEY = "shuffle";

    /**
     * Key to store the number of neurons per hidden layer.
     */
    public static final String NRHNEURONS_KEY = "nrhiddenneurons";

    /**
     * Key to store the class column.
     */
    public static final String CLASSCOL_KEY = "classcol";

    /**
     * Key to store if the random seed should be used.
     *
     */
    public static final String USE_SEED_KEY = "useRandomSeed";

    /**
     * Key to store the random seed.
     *
     */
    public static final String SEED_KEY = "randomSeed";

    /**
     * Key to store learning rate decay rate.
     *
     */
    public static final String LRDECAYRATE_KEY = "lrdecayrate";

    /**
     * Key to store if learning rate decay should be applied.
     *
     */
    public static final String LRDECAY_KEY = "lrdecay";

    private final SettingsModelDoubleBounded m_learnrate =
            new SettingsModelDoubleBounded(LEARNINGRATE_KEY, 0.01, 0.00001, 1);

    private final SettingsModelIntegerBounded m_batchsize =
            new SettingsModelIntegerBounded(BATCHSIZE_KEY, 10, 1, Integer.MAX_VALUE);

    private final SettingsModelIntegerBounded m_lrdecayrate =
            new SettingsModelIntegerBounded(LRDECAYRATE_KEY, 5, 1, Integer.MAX_VALUE);

    private final SettingsModelBoolean m_lrdecay =
            new SettingsModelBoolean(/* config-name: */SimpleMlpNodeModel.LRDECAY_KEY, /* default */false);

    /*
     * Number of iterations.
     */
    private final SettingsModelIntegerBounded m_nrIterations =
            new SettingsModelIntegerBounded(/* config-name: */SimpleMlpNodeModel.MAXITER_KEY,
                    /* default */DEFAULTITERATIONS, /* min: */1, /* max: */SimpleMlpNodeModel.MAXNRITERATIONS);

    /*
     * Number of hidden neurons per layer.
     */
    private final SettingsModelIntegerBounded m_nrHiddenNeuronsperLayer =
            new SettingsModelIntegerBounded(/* config-name: */SimpleMlpNodeModel.NRHNEURONS_KEY,
                    /* default */DEFAULTNEURONSPERLAYER, /* min: */1, /* max: */100);

    /*
     * The class column.
     */
    private final SettingsModelString m_classcol =
            new SettingsModelString(/* config-name: */SimpleMlpNodeModel.CLASSCOL_KEY, null);

    /*
     * Flag whether to ignore missing values
     */
    private final SettingsModelBoolean m_ignoreMV =
            new SettingsModelBoolean(/* config-name: */SimpleMlpNodeModel.IGNOREMV_KEY, /* default */false);

    private final SettingsModelBoolean m_shuffle =
            new SettingsModelBoolean(/* config-name: */SimpleMlpNodeModel.SHUFFLE_KEY, /* default */false);

    private final SettingsModelBoolean m_useRandomSeed = new SettingsModelBoolean(USE_SEED_KEY, false);

    private final SettingsModelInteger m_randomSeed =
            new SettingsModelInteger(SEED_KEY, (int)(2 * (Math.random() - 0.5) * Integer.MAX_VALUE));

    /*
     * Flag for regression
     */
    private boolean m_regression;

    /*
     * The internal Neural Network.
     */
    private SimpleMlp m_mlp;

    /*
     * Used to plot the error.
     */
    // private ErrorPlot m_errorplot;
    /*
     * The error values at each iteration
     */
    private double[] m_errors;

    private HashMap<DataCell, Integer> classMap;

    private int m_nrInputs;

    /**
     * The SimpleMlpModel has 2 inputs, one for the training examples and one for the test ones. The output is the
     * prediction of the trained neural network.
     *
     * @param pmmlInEnabled if true the node has an optional PMML input port
     * @since 3.0
     *
     */
    public SimpleMlpNodeModel() {
        super(new PortType[]{BufferedDataTable.TYPE, BufferedDataTable.TYPE}, new PortType[]{BufferedDataTable.TYPE});
    }

    /**
     * returns null.
     *
     * {@inheritDoc}
     */
    @Override
    protected DataTableSpec[] configure(final DataTableSpec[] inSpecs) throws InvalidSettingsException {
        m_nrInputs = (inSpecs[0].getNumColumns() - 1);
        if (m_classcol.getStringValue() != null) {
            boolean classcolinspec = false;
            for (DataColumnSpec colspec : inSpecs[0]) {
                if (!(colspec.getName().toString().compareTo(m_classcol.getStringValue()) == 0)) {
                    if (!colspec.getType().isCompatible(DoubleValue.class)) {
                        throw new InvalidSettingsException("Only double columns for input");
                    } else {
                        DataColumnDomain domain = colspec.getDomain();
                        if (domain.hasBounds()) {
                            double lower = ((DoubleValue)domain.getLowerBound()).getDoubleValue();
                            double upper = ((DoubleValue)domain.getUpperBound()).getDoubleValue();
                            if (lower < 0 || upper > 1) {
                                setWarningMessage("Input data not normalized." + " Please consider using the "
                                        + "Normalizer Node first.");
                            }
                        }
                    }
                } else {
                    classcolinspec = true;
                    // check for regression
                    if (colspec.getType().isCompatible(DoubleValue.class)) {
                        // check if the values are in range [0,1]
                        DataColumnDomain domain = colspec.getDomain();
                        if (domain.hasBounds()) {
                            double lower = ((DoubleValue)domain.getLowerBound()).getDoubleValue();
                            double upper = ((DoubleValue)domain.getUpperBound()).getDoubleValue();
                            if (lower < 0 || upper > 1) {
                                throw new InvalidSettingsException("Domain range for regression in column "
                                        + colspec.getName() + " not in range [0,1]");
                            }
                        }
                        m_regression = true;
                    } else {
                        m_regression = false;
                    }
                }
            }
            if (!classcolinspec) {
                throw new InvalidSettingsException(
                        "Class column " + m_classcol.getStringValue() + " not found in DataTableSpec");
            }
            ColumnRearranger cr = createColumnRearranger(inSpecs[1]);
            return new DataTableSpec[]{cr.createSpec()};
        } else {
            throw new InvalidSettingsException("Class column not set");
        }
    }

    /**
     * The execution consists of three steps:
     * <ol>
     * <li>A neural network is build with the inputs and outputs according to the input datatable, number of hidden
     * layers as specified.</li>
     * <li>Input DataTables are converted into double-arrays so they can be attached to the neural net.</li>
     * <li>The neural net is trained.</li>
     * </ol>
     *
     * {@inheritDoc}
     */
    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData, final ExecutionContext exec)
            throws Exception {
        // If class column is not set, it is the last column.
        DataTableSpec posSpec = inData[0].getSpec();
        if (m_classcol.getStringValue() == null) {
            m_classcol.setStringValue(posSpec.getColumnSpec(posSpec.getNumColumns() - 1).getName());
        }
        // Determine the number of inputs and the number of outputs. Make also
        // sure that the inputs are double values.
        int nrInputs = 0;
        int nrOutputs = 0;
        HashMap<String, Integer> inputmap = new HashMap<String, Integer>();
        classMap = new HashMap<DataCell, Integer>();
        for (DataColumnSpec colspec : posSpec) {
            // check for class column
            if (colspec.getName().toString().compareTo(m_classcol.getStringValue()) == 0) {
                if (colspec.getType().isCompatible(DoubleValue.class)) {
                    // check if the values are in range [0,1]
                    DataColumnDomain domain = colspec.getDomain();
                    if (domain.hasBounds()) {
                        double lower = ((DoubleValue)domain.getLowerBound()).getDoubleValue();
                        double upper = ((DoubleValue)domain.getUpperBound()).getDoubleValue();
                        if (lower < 0 || upper > 1) {
                            throw new InvalidSettingsException("Domain range for regression in column "
                                    + colspec.getName() + " not in range [0,1]");
                        }
                    }
                    nrOutputs = 1;
                    classMap = new HashMap<DataCell, Integer>();
                    classMap.put(new StringCell(colspec.getName()), 0);
                    m_regression = true;
                } else {
                    m_regression = false;
                    DataColumnDomain domain = colspec.getDomain();
                    if (domain.hasValues()) {
                        Set<DataCell> allvalues = domain.getValues();
                        int outputneuron = 0;
                        classMap = new HashMap<DataCell, Integer>();
                        for (DataCell value : allvalues) {
                            classMap.put(value, outputneuron);
                            outputneuron++;
                        }
                        nrOutputs = allvalues.size();
                    } else {
                        throw new Exception(
                                "Could not find domain values in" + "nominal column " + colspec.getName().toString());
                    }
                }
            } else {
                if (!colspec.getType().isCompatible(DoubleValue.class)) {
                    throw new Exception("Only double columns for input");
                }
                inputmap.put(colspec.getName(), nrInputs);
                nrInputs++;
            }
        }

        Random random = new Random();
        if (m_useRandomSeed.getBooleanValue()) {
            random.setSeed(m_randomSeed.getIntValue());
        }

        if (m_lmode.getStringValue().contentEquals("Online")) {
            m_mlp = new SimpleMlpOnline(nrInputs, m_nrHiddenNeuronsperLayer.getIntValue(), nrOutputs, random,
                    m_learnrate.getDoubleValue(), m_shuffle.getBooleanValue(), m_lrdecayrate.getIntValue(),
                    m_lrdecay.getBooleanValue());
        } else if (m_lmode.getStringValue().contentEquals("Batch")) {
            m_mlp = new SimpleMlpBatch(nrInputs, m_nrHiddenNeuronsperLayer.getIntValue(), nrOutputs, random,
                    m_learnrate.getDoubleValue(), m_shuffle.getBooleanValue(), m_lrdecayrate.getIntValue(),
                    m_lrdecay.getBooleanValue());
        }

        else if (m_lmode.getStringValue().contentEquals("Mini-batch")) {
            m_mlp = new SimpleMlpMini(nrInputs, m_nrHiddenNeuronsperLayer.getIntValue(), nrOutputs, random,
                    m_learnrate.getDoubleValue(), m_batchsize.getIntValue(), m_nrIterations.getIntValue(),
                    m_shuffle.getBooleanValue(), m_lrdecayrate.getIntValue(), m_lrdecay.getBooleanValue());
        }

        int classColNr = posSpec.findColumnIndex(m_classcol.getStringValue());
        List<Double[]> samples = new ArrayList<Double[]>();
        List<Double[]> outputs = new ArrayList<Double[]>();
        Double[] sample = new Double[nrInputs];
        Double[] output = new Double[nrOutputs];
        final RowIterator rowIt = inData[0].iterator();
        int rowcounter = 0;
        while (rowIt.hasNext()) {
            boolean add = true;
            output = new Double[nrOutputs];
            sample = new Double[nrInputs];
            DataRow row = rowIt.next();
            int nrCells = row.getNumCells();
            int index = 0;
            for (int i = 0; i < nrCells; i++) {
                if (i != classColNr) {
                    if (!row.getCell(i).isMissing()) {
                        DoubleValue dc = (DoubleValue)row.getCell(i);
                        sample[index] = dc.getDoubleValue();
                        index++;
                    } else {
                        if (m_ignoreMV.getBooleanValue()) {
                            add = false;
                            break;
                        } else {
                            throw new Exception("Missing values in input" + " datatable");
                        }
                    }
                } else {
                    if (row.getCell(i).isMissing()) {
                        add = false;
                        if (!m_ignoreMV.getBooleanValue()) {
                            throw new Exception("Missing value in class" + " column");
                        }
                        break;
                    }
                    if (m_regression) {
                        DoubleValue dc = (DoubleValue)row.getCell(i);
                        output[0] = dc.getDoubleValue();
                    } else {
                        for (int j = 0; j < nrOutputs; j++) {
                            if (classMap.get(row.getCell(i)) == j) {
                                output[j] = new Double(1.0);
                            } else {
                                output[j] = new Double(0.0);
                            }
                        }
                    }
                }
            }
            if (add) {
                samples.add(sample);
                outputs.add(output);
                rowcounter++;
            }
        }
        Double[][] samplesarr = new Double[rowcounter][nrInputs];
        Double[][] outputsarr = new Double[rowcounter][nrOutputs];
        for (int i = 0; i < samplesarr.length; i++) {
            samplesarr[i] = samples.get(i);
            outputsarr[i] = outputs.get(i);
        }
        // train NN
        m_errors = new double[m_nrIterations.getIntValue()];
        for (int iteration = 0; iteration < m_nrIterations.getIntValue(); iteration++) {
            exec.setProgress((double)iteration / (double)m_nrIterations.getIntValue(), "Iteration " + iteration);
            m_mlp.trainNetwork(samplesarr, outputsarr, iteration);
            double error = 0;
            for (int j = 0; j < outputsarr.length; j++) {
                Double[] myoutput = m_mlp.passExample(samplesarr[j]);
                for (int o = 0; o < outputsarr[0].length; o++) {
                    error += (myoutput[o] - outputsarr[j][o]) * (myoutput[o] - outputsarr[j][o]);
                }
            }
            m_errors[iteration] = error;
            if (error < 0.2) {
                break;
            }
            exec.checkCanceled();

        }

        BufferedDataTable table =
                exec.createColumnRearrangeTable(inData[1], createColumnRearranger(inData[1].getDataTableSpec()), exec);
        return new BufferedDataTable[]{table};

    }

    private ColumnRearranger createColumnRearranger(final DataTableSpec spec) throws InvalidSettingsException {
        // check user settings against input spec here
        // fail with InvalidSettingsException if invalid

        for (DataColumnSpec colspec : spec) {
            // check for class column
            if (colspec.getName().toString().compareTo(m_classcol.getStringValue()) == 0) {
                throw new InvalidSettingsException("Class column must not be presented for predicion!");
            }
        }
        if (spec.getNumColumns() != m_nrInputs) {
            throw new InvalidSettingsException("Number of columns for prediction and learning are different!");
        }

        // the following code appends a single column
        DataColumnSpec appendSpec;
        if (m_regression) {
            appendSpec = new DataColumnSpecCreator("Prediction(" + m_classcol.getStringValue() + ")", DoubleCell.TYPE)
                    .createSpec();
        } else {
            appendSpec = new DataColumnSpecCreator("Prediction(" + m_classcol.getStringValue() + ")", StringCell.TYPE)
                    .createSpec();
        }
        SingleCellFactory factory = new SingleCellFactory(appendSpec) {
            @Override
            public DataCell getCell(final DataRow row) {
                Double[] sample = new Double[row.getNumCells()];
                DataCell res = null;
                for (int i = 0; i < row.getNumCells(); i++) {
                    sample[i] = Double.parseDouble(row.getCell(i).toString());
                }
                if (m_regression) {
                    res = new DoubleCell(m_mlp.predict(sample)[0]);
                } else {
                    Double[] temp = m_mlp.predict(sample);
                    Double max = temp[0];
                    Integer indx = 0;
                    for (int j = 1; j < temp.length; j++) {
                        if (temp[j] > max) {
                            max = temp[j];
                            indx = j;
                        }
                    }
                    for (Entry<DataCell, Integer> entry : classMap.entrySet()) {
                        if (indx == entry.getValue()) {
                            res = new StringCell(entry.getKey().toString());
                            break;
                        }
                    }
                }
                return res;

            }
        };
        ColumnRearranger result = new ColumnRearranger(spec);
        result.append(factory);
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void reset() {
        m_errors = null;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        m_nrIterations.saveSettingsTo(settings);
        m_nrHiddenNeuronsperLayer.saveSettingsTo(settings);
        m_classcol.saveSettingsTo(settings);
        m_ignoreMV.saveSettingsTo(settings);
        m_useRandomSeed.saveSettingsTo(settings);
        m_randomSeed.saveSettingsTo(settings);
        m_lmode.saveSettingsTo(settings);
        m_learnrate.saveSettingsTo(settings);
        m_batchsize.saveSettingsTo(settings);
        m_shuffle.saveSettingsTo(settings);
        m_lrdecay.saveSettingsTo(settings);
        m_lrdecayrate.saveSettingsTo(settings);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_nrIterations.validateSettings(settings);
        m_nrHiddenNeuronsperLayer.validateSettings(settings);
        m_classcol.validateSettings(settings);
        m_ignoreMV.validateSettings(settings);
        m_lmode.validateSettings(settings);
        m_learnrate.validateSettings(settings);
        m_batchsize.validateSettings(settings);
        m_shuffle.validateSettings(settings);
        m_lrdecay.validateSettings(settings);
        m_lrdecayrate.validateSettings(settings);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {

        m_nrIterations.loadSettingsFrom(settings);
        m_nrHiddenNeuronsperLayer.loadSettingsFrom(settings);
        m_classcol.loadSettingsFrom(settings);
        m_ignoreMV.loadSettingsFrom(settings);
        m_lmode.loadSettingsFrom(settings);
        m_learnrate.loadSettingsFrom(settings);
        m_batchsize.loadSettingsFrom(settings);
        m_shuffle.loadSettingsFrom(settings);
        m_lrdecayrate.loadSettingsFrom(settings);
        m_lrdecay.loadSettingsFrom(settings);

        try {
            m_useRandomSeed.loadSettingsFrom(settings);
        } catch (InvalidSettingsException ex) {
            // use current/default value
        }
        try {
            m_randomSeed.loadSettingsFrom(settings);
        } catch (InvalidSettingsException ex) {
            // use current/default value
        }
    }

    /**
     * @return error plot.
     */
    public double[] getErrors() {
        return m_errors;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadInternals(final File internDir, final ExecutionMonitor exec) throws IOException {
        File f = new File(internDir, "SimplMLP");
        ObjectInputStream in = new ObjectInputStream(new FileInputStream(f));
        int iterations = in.readInt();
        m_errors = new double[iterations];
        for (int i = 0; i < iterations; i++) {
            m_errors[i] = in.readDouble();
            exec.setProgress((double)i / (double)iterations);
        }
        in.close();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveInternals(final File internDir, final ExecutionMonitor exec) throws IOException {
        File f = new File(internDir, "SimplMLP");
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(f));
        int iterations = m_errors.length;
        out.writeInt(iterations);
        for (int i = 0; i < iterations; i++) {
            out.writeDouble(m_errors[i]);
            exec.setProgress((double)i / (double)iterations);
        }
        out.close();
    }
}
