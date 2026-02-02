package parking;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import javax.swing.table.DefaultTableModel;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;

import java.awt.print.PrinterException;
import java.awt.print.PrinterJob;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Parking Lot Management System (Beautiful + Informative Swing GUI)
 *
 * ✅ Park Vehicle -> assigns slot + stores entry time
 * ✅ Exit Vehicle -> calculates price based on time parked (hourly billing)
 * ✅ Receipt -> shows receipt on exit + saves receipt file + optional print
 * ✅ View Status -> shows occupied/free slots + entry time in a table
 * ✅ View Collection -> shows totals from payments history
 *
 * FILES:
 * - parking.txt   (currently parked)   slot,vehicleNo,type,entryMillis
 * - payments.txt  (payment history)    vehicleNo,type,entryMillis,exitMillis,hours,amount
 * - receipts/     (saved receipts)     receipt_<vehicleNo>_<yyyyMMdd_HHmmss>.txt
 */
public class ParkingLotGUI extends JFrame {

    // UI
    private JTextField vehicleNoField;
    private JComboBox<String> typeBox;
    private JLabel filePathLabel;
    private JLabel freeSlotsLabel;
    private JLabel parkedCountLabel;
    private JLabel totalCollectedLabel;
    private JTextArea activityLog;

    // Table
    private DefaultTableModel tableModel;
    private JTable statusTable;

    // Files / config
    static final String PARKING_FILE = "parking.txt";
    static final String PAYMENT_FILE = "payments.txt";
    static final String RECEIPT_DIR = "receipts";
    static final int TOTAL_SLOTS = 10;

    // Hourly pricing
    static final int RATE_BIKE_PER_HOUR = 20;
    static final int RATE_CAR_PER_HOUR = 50;
    static final int RATE_TRUCK_PER_HOUR = 80;

    // Minimum billing
    static final int MIN_HOURS = 1;

    private final SimpleDateFormat sdf = new SimpleDateFormat("dd-MM-yyyy HH:mm:ss");
    private final SimpleDateFormat receiptStamp = new SimpleDateFormat("yyyyMMdd_HHmmss");
    private final DecimalFormat df = new DecimalFormat("0.00");

    public ParkingLotGUI() {
        // Window
        setTitle("Parking Lot Management System • Time-Based Billing");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setMinimumSize(new Dimension(950, 600));
        setLocationRelativeTo(null);

        // Create files/dirs
        createFilesAndDirs();

        // Use a nicer font globally
        setUIFont(new Font("Segoe UI", Font.PLAIN, 14));

        // Layout
        setLayout(new BorderLayout(12, 12));
        ((JComponent) getContentPane()).setBorder(new EmptyBorder(12, 12, 12, 12));

        add(buildHeaderPanel(), BorderLayout.NORTH);
        add(buildCenterPanel(), BorderLayout.CENTER);
        add(buildFooterPanel(), BorderLayout.SOUTH);

        // Initial refresh
        refreshDashboard();
        refreshStatusTable();
        log("System started. Ready.");

        setVisible(true);
    }

    // ---------- CUSTOM EXCEPTION ----------
    static class ParkingFullException extends Exception {
        public ParkingFullException(String msg) { super(msg); }
    }

    // -------------------- UI PANELS --------------------
    private JPanel buildHeaderPanel() {
        JPanel header = new JPanel(new BorderLayout(10, 10));
        header.setBorder(new TitledBorder("Dashboard"));

        JPanel left = new JPanel(new FlowLayout(FlowLayout.LEFT, 12, 6));
        JLabel title = new JLabel("Parking Lot • Time Billing");
        title.setFont(new Font("Segoe UI", Font.BOLD, 18));
        left.add(title);

        filePathLabel = new JLabel("File: " + new File(PARKING_FILE).getAbsolutePath());
        filePathLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        left.add(filePathLabel);

        JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 12, 6));
        freeSlotsLabel = makeStatLabel("Free Slots: 0");
        parkedCountLabel = makeStatLabel("Parked: 0");
        totalCollectedLabel = makeStatLabel("Collected: ₹0");
        right.add(freeSlotsLabel);
        right.add(parkedCountLabel);
        right.add(totalCollectedLabel);

        header.add(left, BorderLayout.WEST);
        header.add(right, BorderLayout.EAST);
        return header;
    }

    private JPanel buildCenterPanel() {
        JPanel center = new JPanel(new BorderLayout(12, 12));

        // Left side: Form + Buttons + Activity log
        JPanel left = new JPanel(new BorderLayout(12, 12));
        left.setBorder(new TitledBorder("Operations"));

        left.add(buildFormPanel(), BorderLayout.NORTH);
        left.add(buildButtonsPanel(), BorderLayout.CENTER);
        left.add(buildLogPanel(), BorderLayout.SOUTH);

        // Right side: Status table
        JPanel right = new JPanel(new BorderLayout(12, 12));
        right.setBorder(new TitledBorder("Live Status"));

        right.add(buildTablePanel(), BorderLayout.CENTER);

        center.add(left, BorderLayout.WEST);
        center.add(right, BorderLayout.CENTER);

        return center;
    }

    private JPanel buildFormPanel() {
        JPanel form = new JPanel(new FlowLayout(FlowLayout.LEFT, 12, 10));

        JLabel vLabel = new JLabel("Vehicle No:");
        vehicleNoField = new JTextField(14);

        JLabel tLabel = new JLabel("Type:");
        typeBox = new JComboBox<>(new String[]{"BIKE", "CAR", "TRUCK"});

        form.add(vLabel);
        form.add(vehicleNoField);
        form.add(tLabel);
        form.add(typeBox);

        return form;
    }

    private JPanel buildButtonsPanel() {
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.LEFT, 12, 10));

        JButton parkBtn = makePrimaryButton("Park Vehicle");
        JButton exitBtn = makePrimaryButton("Exit Vehicle");

        JButton refreshBtn = new JButton("Refresh");
        JButton collectionBtn = new JButton("View Collection");

        parkBtn.addActionListener(e -> parkVehicle());
        exitBtn.addActionListener(e -> exitVehicle());
        refreshBtn.addActionListener(e -> {
            refreshDashboard();
            refreshStatusTable();
            log("Refreshed status.");
        });
        collectionBtn.addActionListener(e -> showCollectionDialog());

        buttons.add(parkBtn);
        buttons.add(exitBtn);
        buttons.add(refreshBtn);
        buttons.add(collectionBtn);

        JPanel info = new JPanel(new FlowLayout(FlowLayout.LEFT, 12, 10));
        info.setBorder(new TitledBorder("Rates"));
        info.add(new JLabel("BIKE: ₹" + RATE_BIKE_PER_HOUR + "/hr"));
        info.add(new JLabel("CAR: ₹" + RATE_CAR_PER_HOUR + "/hr"));
        info.add(new JLabel("TRUCK: ₹" + RATE_TRUCK_PER_HOUR + "/hr"));
        info.add(new JLabel("Min Billing: " + MIN_HOURS + " hour"));

        JPanel wrap = new JPanel(new BorderLayout());
        wrap.add(buttons, BorderLayout.NORTH);
        wrap.add(info, BorderLayout.SOUTH);
        return wrap;
    }

    private JPanel buildLogPanel() {
        JPanel logPanel = new JPanel(new BorderLayout(8, 8));
        logPanel.setBorder(new TitledBorder("Activity Log"));

        activityLog = new JTextArea(8, 32);
        activityLog.setEditable(false);
        activityLog.setLineWrap(true);
        activityLog.setWrapStyleWord(true);

        JScrollPane sp = new JScrollPane(activityLog);
        sp.setPreferredSize(new Dimension(380, 180));

        logPanel.add(sp, BorderLayout.CENTER);
        return logPanel;
    }

    private JPanel buildTablePanel() {
        tableModel = new DefaultTableModel(
                new Object[]{"Slot", "Vehicle No", "Type", "Entry Time", "Current Hours", "Est. Amount (₹)"},
                0
        ) {
            @Override
            public boolean isCellEditable(int row, int col) { return false; }
        };

        statusTable = new JTable(tableModel);
        statusTable.setRowHeight(24);
        statusTable.getTableHeader().setFont(new Font("Segoe UI", Font.BOLD, 13));

        JScrollPane sp = new JScrollPane(statusTable);
        return new JPanel(new BorderLayout()) {{
            add(sp, BorderLayout.CENTER);
        }};
    }

    private JPanel buildFooterPanel() {
        JPanel footer = new JPanel(new BorderLayout(8, 8));
        footer.setBorder(new EmptyBorder(4, 4, 4, 4));
        JLabel tip = new JLabel("Tip: Enter vehicle number (ex: KA01AB1234). Park → Exit to generate payment + receipt.");
        tip.setFont(new Font("Segoe UI", Font.ITALIC, 12));
        footer.add(tip, BorderLayout.WEST);
        return footer;
    }

    // -------------------- CORE LOGIC --------------------
    void createFilesAndDirs() {
        try {
            new File(PARKING_FILE).createNewFile();
            new File(PAYMENT_FILE).createNewFile();
            File rdir = new File(RECEIPT_DIR);
            if (!rdir.exists()) rdir.mkdirs();
        } catch (IOException e) {
            JOptionPane.showMessageDialog(this, "File creation error");
        }
    }

    void parkVehicle() {
        String vehicleNo = vehicleNoField.getText().trim();
        String type = (String) typeBox.getSelectedItem();

        if (vehicleNo.isEmpty()) {
            warn("Enter Vehicle Number");
            return;
        }

        if (isAlreadyParked(vehicleNo)) {
            warn("This vehicle is already parked");
            return;
        }

        try {
            int slot = getNextFreeSlot(); // may throw ParkingFullException
            long entryMillis = System.currentTimeMillis();

            try (FileWriter fw = new FileWriter(PARKING_FILE, true)) {
                fw.write(slot + "," + vehicleNo + "," + type + "," + entryMillis + "\n");
            }

            info("Parked Successfully!\nSlot: " + slot + "\nEntry: " + sdf.format(new Date(entryMillis))
                    + "\nRate: ₹" + getRatePerHour(type) + "/hour");
            log("PARKED: " + vehicleNo + " (" + type + ") in Slot " + slot);

            vehicleNoField.setText("");
            refreshDashboard();
            refreshStatusTable();

        } catch (ParkingFullException ex) {
            warn(ex.getMessage());
            log("FAILED PARK: " + vehicleNo + " (Parking Full)");
        } catch (IOException ex) {
            warn("Error writing to parking file");
        }
    }

    void exitVehicle() {
        String vehicleNo = vehicleNoField.getText().trim();

        if (vehicleNo.isEmpty()) {
            warn("Enter Vehicle Number");
            return;
        }

        List<String> remainingLines = new ArrayList<>();
        boolean found = false;

        String typeFound = "";
        int slotFound = -1;
        long entryMillisFound = 0L;

        // Read parking file and remove vehicle
        try (BufferedReader br = new BufferedReader(new FileReader(PARKING_FILE))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split(",");
                if (parts.length < 4) {
                    remainingLines.add(line);
                    continue;
                }

                int slot = Integer.parseInt(parts[0].trim());
                String fileVehicleNo = parts[1].trim();
                String type = parts[2].trim();
                long entryMillis = Long.parseLong(parts[3].trim());

                if (fileVehicleNo.equalsIgnoreCase(vehicleNo)) {
                    found = true;
                    slotFound = slot;
                    typeFound = type;
                    entryMillisFound = entryMillis;
                } else {
                    remainingLines.add(line);
                }
            }
        } catch (IOException e) {
            warn("Error reading parking file");
            return;
        }

        if (!found) {
            warn("Vehicle not found in parking lot");
            log("FAILED EXIT: " + vehicleNo + " (Not Found)");
            return;
        }

        long exitMillis = System.currentTimeMillis();
        int hours = calculateBillableHours(entryMillisFound, exitMillis);
        int rate = getRatePerHour(typeFound);
        int amount = hours * rate;

        // Rewrite parking file
        try (FileWriter fw = new FileWriter(PARKING_FILE, false)) {
            for (String l : remainingLines) fw.write(l + "\n");
        } catch (IOException e) {
            warn("Error updating parking file");
            return;
        }

        // Append payment history
        try (FileWriter fw = new FileWriter(PAYMENT_FILE, true)) {
            fw.write(vehicleNo + "," + typeFound + "," + entryMillisFound + "," + exitMillis + "," + hours + "," + amount + "\n");
        } catch (IOException e) {
            warn("Error writing payment file");
            return;
        }

        // ✅ RECEIPT FEATURE
        Receipt receipt = new Receipt(
                vehicleNo, typeFound, slotFound,
                entryMillisFound, exitMillis, hours, rate, amount
        );

        String receiptText = buildReceiptText(receipt);

        File savedReceipt = null;
        try {
            savedReceipt = saveReceiptToFile(receiptText, receipt);
        } catch (IOException e) {
            // Still show receipt even if saving fails
            log("RECEIPT SAVE FAILED for " + vehicleNo + " (" + e.getMessage() + ")");
        }

        // Existing info dialog (quick)
        info("Exited Successfully!\nSlot: " + slotFound
                + "\nType: " + typeFound
                + "\nEntry: " + sdf.format(new Date(entryMillisFound))
                + "\nExit : " + sdf.format(new Date(exitMillis))
                + "\nBilled Hours: " + hours
                + "\nRate: ₹" + rate + "/hour"
                + "\nTotal Amount: ₹" + amount);

        // Show detailed receipt dialog (with Save path + Print)
        showReceiptDialog(receiptText, savedReceipt);

        log("EXITED: " + vehicleNo + " (" + typeFound + ") | Slot " + slotFound + " | ₹" + amount);
        vehicleNoField.setText("");

        refreshDashboard();
        refreshStatusTable();
    }

    // -------------------- RECEIPT HELPERS --------------------
    static class Receipt {
        final String vehicleNo;
        final String type;
        final int slot;
        final long entryMillis;
        final long exitMillis;
        final int billedHours;
        final int ratePerHour;
        final int amount;

        Receipt(String vehicleNo, String type, int slot, long entryMillis, long exitMillis,
                int billedHours, int ratePerHour, int amount) {
            this.vehicleNo = vehicleNo;
            this.type = type;
            this.slot = slot;
            this.entryMillis = entryMillis;
            this.exitMillis = exitMillis;
            this.billedHours = billedHours;
            this.ratePerHour = ratePerHour;
            this.amount = amount;
        }
    }

    private String buildReceiptText(Receipt r) {
        long diff = r.exitMillis - r.entryMillis;
        if (diff < 0) diff = 0;

        double minutes = diff / 60000.0;
        double hoursExact = diff / (3600000.0);

        String line = "------------------------------------------------------------";
        return ""
                + "                    PARKING PAYMENT RECEIPT\n"
                + line + "\n"
                + "Receipt Time : " + sdf.format(new Date(r.exitMillis)) + "\n"
                + "Vehicle No   : " + r.vehicleNo + "\n"
                + "Vehicle Type : " + r.type + "\n"
                + "Slot No      : " + r.slot + "\n"
                + line + "\n"
                + "Entry Time   : " + sdf.format(new Date(r.entryMillis)) + "\n"
                + "Exit Time    : " + sdf.format(new Date(r.exitMillis)) + "\n"
                + "Duration     : " + df.format(minutes) + " minutes (" + df.format(hoursExact) + " hours)\n"
                + "Billed Hours : " + r.billedHours + " (ceil, min " + MIN_HOURS + ")\n"
                + "Rate         : ₹" + r.ratePerHour + " / hour\n"
                + line + "\n"
                + "TOTAL AMOUNT : ₹" + r.amount + "\n"
                + line + "\n"
                + "Thank you! Drive safe.\n";
    }

    private File saveReceiptToFile(String receiptText, Receipt r) throws IOException {
        File dir = new File(RECEIPT_DIR);
        if (!dir.exists()) dir.mkdirs();

        String safeVehicle = r.vehicleNo.replaceAll("[^a-zA-Z0-9_-]", "_");
        String name = "receipt_" + safeVehicle + "_" + receiptStamp.format(new Date(r.exitMillis)) + ".txt";
        File out = new File(dir, name);

        try (FileWriter fw = new FileWriter(out, false)) {
            fw.write(receiptText);
        }

        log("RECEIPT SAVED: " + out.getAbsolutePath());
        return out;
    }

    private void showReceiptDialog(String receiptText, File savedReceipt) {
        JTextArea area = new JTextArea(receiptText, 22, 56);
        area.setEditable(false);
        area.setFont(new Font("Consolas", Font.PLAIN, 13));
        area.setCaretPosition(0);

        JScrollPane sp = new JScrollPane(area);
        sp.setPreferredSize(new Dimension(720, 420));

        String footer = (savedReceipt != null)
                ? ("\nSaved to: " + savedReceipt.getAbsolutePath())
                : ("\nSaved to: (save failed — check permissions)");

        Object[] options = {"Print", "Close"};
        int choice = JOptionPane.showOptionDialog(
                this,
                new Object[]{sp, new JLabel(footer)},
                "Receipt",
                JOptionPane.YES_NO_OPTION,
                JOptionPane.INFORMATION_MESSAGE,
                null,
                options,
                options[1]
        );

        if (choice == 0) { // Print
            try {
                printTextArea(area);
                log("RECEIPT PRINTED");
            } catch (PrinterException ex) {
                warn("Printing failed: " + ex.getMessage());
                log("RECEIPT PRINT FAILED: " + ex.getMessage());
            }
        }
    }

    private void printTextArea(JTextArea area) throws PrinterException {
        PrinterJob job = PrinterJob.getPrinterJob();
        job.setJobName("Parking Receipt");
        // Simple way: use JTextComponent print API
        boolean ok = area.print();
        if (!ok) throw new PrinterException("User cancelled printing.");
    }

    // -------------------- DASHBOARD + TABLE --------------------
    void refreshDashboard() {
        int parked = countCurrentlyParked();
        int free = TOTAL_SLOTS - parked;
        int total = totalCollectedFromPayments();

        freeSlotsLabel.setText("Free Slots: " + Math.max(free, 0));
        parkedCountLabel.setText("Parked: " + parked);
        totalCollectedLabel.setText("Collected: ₹" + total);
    }

    void refreshStatusTable() {
        tableModel.setRowCount(0);

        Map<Integer, Record> map = new TreeMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(PARKING_FILE))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] p = line.split(",");
                if (p.length < 4) continue;

                int slot = Integer.parseInt(p[0].trim());
                String vno = p[1].trim();
                String type = p[2].trim();
                long entryMillis = Long.parseLong(p[3].trim());

                map.put(slot, new Record(slot, vno, type, entryMillis));
            }
        } catch (IOException e) {
            warn("Error reading parking file");
            return;
        }

        long now = System.currentTimeMillis();

        for (int slot = 1; slot <= TOTAL_SLOTS; slot++) {
            Record r = map.get(slot);
            if (r == null) {
                tableModel.addRow(new Object[]{slot, "-", "-", "-", "-", "-"});
            } else {
                int hours = calculateBillableHours(r.entryMillis, now);
                int est = hours * getRatePerHour(r.type);
                tableModel.addRow(new Object[]{
                        r.slot,
                        r.vehicleNo,
                        r.type,
                        sdf.format(new Date(r.entryMillis)),
                        hours,
                        est
                });
            }
        }
    }

    // Dialog for collection info
    void showCollectionDialog() {
        int parked = countCurrentlyParked();
        int total = totalCollectedFromPayments();

        String msg = "COLLECTION SUMMARY\n\n"
                + "Currently Parked Vehicles: " + parked + "\n"
                + "Total Collected (from exited vehicles): ₹" + total + "\n\n"
                + "Files:\n"
                + "- " + new File(PARKING_FILE).getAbsolutePath() + "\n"
                + "- " + new File(PAYMENT_FILE).getAbsolutePath() + "\n"
                + "- " + new File(RECEIPT_DIR).getAbsolutePath() + " (receipts folder)";

        JOptionPane.showMessageDialog(this, msg, "Collection", JOptionPane.INFORMATION_MESSAGE);
        log("Viewed collection summary.");
    }

    // -------------------- HELPERS --------------------
    boolean isAlreadyParked(String vehicleNo) {
        try (BufferedReader br = new BufferedReader(new FileReader(PARKING_FILE))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] p = line.split(",");
                if (p.length >= 2 && p[1].trim().equalsIgnoreCase(vehicleNo)) return true;
            }
        } catch (IOException e) {
            // ignore
        }
        return false;
    }

    int getNextFreeSlot() throws ParkingFullException {
        boolean[] used = new boolean[TOTAL_SLOTS + 1];

        try (BufferedReader br = new BufferedReader(new FileReader(PARKING_FILE))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] p = line.split(",");
                if (p.length >= 1) {
                    int slot = Integer.parseInt(p[0].trim());
                    if (slot >= 1 && slot <= TOTAL_SLOTS) used[slot] = true;
                }
            }
        } catch (IOException e) {
            // treat as empty
        }

        for (int i = 1; i <= TOTAL_SLOTS; i++) {
            if (!used[i]) return i;
        }
        throw new ParkingFullException("Parking Lot is FULL! No slots available.");
    }

    int getRatePerHour(String type) {
        if ("BIKE".equals(type)) return RATE_BIKE_PER_HOUR;
        if ("CAR".equals(type)) return RATE_CAR_PER_HOUR;
        return RATE_TRUCK_PER_HOUR;
    }

    int calculateBillableHours(long entryMillis, long exitMillis) {
        long diff = exitMillis - entryMillis;
        if (diff < 0) diff = 0;

        long hourMs = 60L * 60L * 1000L;
        int hours = (int) ((diff + hourMs - 1) / hourMs); // ceil
        if (hours < MIN_HOURS) hours = MIN_HOURS;
        return hours;
    }

    int countCurrentlyParked() {
        int count = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(PARKING_FILE))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] p = line.split(",");
                if (p.length >= 4) count++;
            }
        } catch (IOException e) {
            // ignore
        }
        return count;
    }

    int totalCollectedFromPayments() {
        int total = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(PAYMENT_FILE))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] p = line.split(",");
                if (p.length >= 6) total += Integer.parseInt(p[5].trim());
            }
        } catch (IOException e) {
            // ignore
        }
        return total;
    }

    // -------------------- SMALL UI HELPERS --------------------
    private JLabel makeStatLabel(String text) {
        JLabel l = new JLabel(text);
        l.setFont(new Font("Segoe UI", Font.BOLD, 14));
        l.setBorder(new EmptyBorder(0, 6, 0, 6));
        return l;
    }

    private JButton makePrimaryButton(String text) {
        JButton b = new JButton(text);
        b.setFont(new Font("Segoe UI", Font.BOLD, 14));
        b.setPreferredSize(new Dimension(150, 34));
        return b;
    }

    private void log(String msg) {
        activityLog.append("• " + sdf.format(new Date()) + "  -  " + msg + "\n");
        activityLog.setCaretPosition(activityLog.getDocument().getLength());
    }

    private void warn(String msg) {
        JOptionPane.showMessageDialog(this, msg, "Warning", JOptionPane.WARNING_MESSAGE);
    }

    private void info(String msg) {
        JOptionPane.showMessageDialog(this, msg, "Info", JOptionPane.INFORMATION_MESSAGE);
    }

    // Set global font for Swing
    private static void setUIFont(Font f) {
        java.util.Enumeration<Object> keys = UIManager.getDefaults().keys();
        while (keys.hasMoreElements()) {
            Object key = keys.nextElement();
            Object value = UIManager.get(key);
            if (value instanceof javax.swing.plaf.FontUIResource) {
                UIManager.put(key, new javax.swing.plaf.FontUIResource(f));
            }
        }
    }

    // Record holder
    static class Record {
        int slot;
        String vehicleNo;
        String type;
        long entryMillis;

        Record(int slot, String vehicleNo, String type, long entryMillis) {
            this.slot = slot;
            this.vehicleNo = vehicleNo;
            this.type = type;
            this.entryMillis = entryMillis;
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(ParkingLotGUI::new);
    }
}
