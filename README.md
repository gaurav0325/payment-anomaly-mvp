# payment-anomaly-mvp

AI/ML Payment Anomaly Detection System for Settlement \& Reconciliation











\# DART File Specifications



\## Overview



The Data Analytics Reconciliation Tool (DART) contains information currently available for collection from FIS in a data file format. DART is available to merchants acquired by FIS.



\## File Types



\### 312 Transaction Reconciliation File

Replaces the legacy Worldpay Transaction Reconciliation file.



\### 313 Funding Reconciliation File  

Replaces the legacy Worldpay Funding Reconciliation Report.



\## Common File Properties



\- \*\*Format\*\*: Standard ASCII CSV format

\- \*\*Fields\*\*: Positional and variable length

\- \*\*Availability\*\*: Files produced daily, 7 days a week, including Bank Holidays

\- \*\*Retention\*\*: Files retained for seven calendar days from creation

\- \*\*Schedule\*\*: Will be confirmed in future communications

\- \*\*Destination\*\*: SFG mailbox



\## File Naming Convention



\- \*\*312 File\*\*: `/Outgoing/WP\_SSSS\_312TRC\_V03\_yyyymmdd\_nnn.CSV`

\- \*\*313 File\*\*: `WP\_ssss\_313PRC\_V03\_yyyymmdd\_nnn.CSV`



Where:

\- `SSSS` = Unique Merchant acronym

\- `yyyymmdd` = Date format

\- `nnn` = Sequential number



---



\## 312 Transaction Reconciliation File Structure



\### Record Type 00 - Header Record



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | 00 – Header Record |

| 2 | Data File Date | 8 | DATE | Data file name run date (YYYYMMDD) |

| 3 | Party ID | 15 | VARCHAR | Transacting Party Identifier |

| 4 | Total Submitted Number of Transactions | 10 | NUM | Total number of transactions submitted |

| 5 | Total Accepted number of transactions | 10 | NUM | Total accepted and will be funded |

| 6 | Total Pending number of transactions | 10 | NUM | Overall totals - pending transactions |

| 7 | Total Rejected number of transactions | 10 | NUM | Overall totals - rejected transactions |

| 8 | Total Transaction Value of Accepted | 23 | NUM | Total value in transaction currency |

| 9 | Total Settlement Value of Accepted | 23 | NUM | Total value in settlement currency |

| 10 | Total Transaction Value of Pending | 23 | NUM | Total value in transaction currency |

| 11 | Total Transaction Value of Rejected | 23 | NUM | Total value in transaction currency |



\### Record Type 01 - Accepted Transactions Record



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | 01 - Accepted transactions information |

| 2 | Transacting Party ID | 15 | VARCHAR | Transacting party identifier |

| 3 | Merchant ID | 15 | CHAR | Merchant ID for transactional activity |

| 4 | Store Reference | 30 | CHAR | Store reference number |

| 5 | Terminal ID | 16 | CHAR | The terminal ID of the transaction |

| 6 | Merchant Classification Code (MCC) | 4 | NUM | Industry classifier code |

| 7 | Transaction Type | 25 | CHAR | Type of transaction submitted to Worldpay |

| 8 | Single Message OCT | 1 | CHAR | Y for Single Message, N for dual message |

| 9 | Settlement Party ID | 15 | VARCHAR | Party ID for settlement |

| 10 | Issuer BIN country | 3 | CHAR | Country card was acquired (ISO alpha) |

| 11 | PAN | 19 | CHAR | Masked PAN (actual or tokenised) |

| 12 | Card Expiry Date | 4 | DATE | Card expiry Date (MMYY) |

| 13 | Transaction Amount | 23 | NUM | Amount in major units with exponent applied |

| 14 | Transaction Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 15 | Transaction Cashback amount | 23 | NUM | Cashback amount in major units |

| 16 | Original Amount (pre DCC) | 23 | NUM | DCC cardholder amount if applicable |

| 17 | Original Amount Currency (pre DCC) | 3 | CHAR | DCC cardholder currency if applicable |

| 18 | Acquired/Processed Flag | 1 | CHAR | Worldpay acquired/processed transaction |

| 19 | Settlement Amount | 23 | NUM | Merchant settlement amount |

| 20 | Settlement Currency | 3 | CHAR | Merchant settlement currency |

| 21 | Transaction Date/Time | 15 | DATE | Format: YYYYMMDD HHMISS |

| 22 | Pricing Segment Code | 5 | CHAR | 5 character pricing segment code |

| 23 | Payee Reference (OTR) | 34 | VARCHAR | Originators transaction reference |

| 24 | Acquirer Reference Number (ARN) | 23 | VARCHAR | Unique identifier for transaction |

| 25 | Scheme Reference | 15 | CHAR | Scheme reference |

| 26 | Worldpay Transaction Source | 1 | CHAR | Source of original transaction |

| 27 | Airline Ticket Number | 255 | CHAR | Airline ticket number if applicable |

| 28 | Authorisation Method | 1 | NUM | Authorisation method |

| 29 | Authorisation Code | 9 | NUM | Authorisation code |

| 30 | Trading Day | 8 | DATE | Transaction date in DDMMYY format |



\### Record Type 02 - Pending Transactions Record



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | 02 - Pending transactions information |

| 2 | Party ID | 15 | VARCHAR | Transacting party Identifier |

| 3 | Merchant ID | 15 | CHAR | Merchant ID for transactional activity |

| 4 | Store Reference | 30 | CHAR | Store reference number |

| 5 | Terminal ID | 16 | CHAR | The terminal ID of the transaction |

| 6 | Merchant Classification Code | 4 | NUM | Industry classifier code |

| 7 | Transaction Type | 25 | CHAR | Type of transaction submitted to FIS |

| 8 | Single Message OCT | 1 | CHAR | Y for Single Message, N for dual message |

| 9 | Cleared/Review | 1 | CHAR | Transaction cleared or under assessment |

| 10 | Issuer BIN country | 3 | CHAR | Country card was acquired (ISO alpha) |

| 11 | PAN | 19 | CHAR | Masked PAN (actual or tokenised) |

| 12 | Card Expiry Date | 4 | DATE | Card expiry Date (MMYY) |

| 13 | Transaction Amount | 23 | NUM | Amount in major units with exponent |

| 14 | Transaction Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 15 | Transaction Cashback Amount | 23 | NUM | Cashback amount in major units |

| 16 | Original Amount (pre DCC) | 23 | NUM | DCC cardholder amount if applicable |

| 17 | Original Amount Currency (pre DCC) | 3 | CHAR | DCC cardholder currency |

| 18 | Transaction Date/Time | 15 | DATE | Format: YYYYMMDD HHMISS |

| 19 | Pricing Segment Code | 5 | CHAR | 5 character pricing segment code |

| 20 | Payee Reference (OTR) | 34 | CHAR | Originators transaction reference |

| 21 | Worldpay Transaction Source | 1 | CHAR | Source of original transaction |

| 22 | Airline Ticket Number | 255 | CHAR | Airline ticket number if applicable |

| 23 | Authorisation Method | 1 | NUM | Authorisation method |

| 24 | Authorisation Code | 9 | NUM | Authorisation code |

| 25 | Trading Day | 8 | DATE | Transaction date in DDMMYY format |



\### Record Type 03 - Rejected Transactions Record



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | 03 - Rejected transactions information |

| 2 | Party ID | 15 | VARCHAR | Transacting Party Identifier |

| 3 | Merchant ID | 15 | CHAR | Merchant ID for transactional activity |

| 4 | Store Reference | 30 | CHAR | Store Reference Number |

| 5 | Terminal ID | 16 | CHAR | The terminal ID of the transaction |

| 6 | Merchant Classification Code | 4 | NUM | Industry classifier code |

| 7 | Transaction Type | 25 | CHAR | Type of transaction submitted to FIS |

| 8 | Single Message OCT | 1 | CHAR | Y for Single Message, N for dual message |

| 9 | Issuer BIN Country | 3 | CHAR | Country card was acquired (ISO alpha) |

| 10 | PAN | 19 | CHAR | Masked PAN (actual or tokenised) |

| 11 | Card Expiry Date | 4 | DATE | Card expiry Date (MMYY) |

| 12 | Transaction Amount | 23 | NUM | Amount in major units with exponent |

| 13 | Transaction currency | 3 | CHAR | Currency in ISO Alpha standards |

| 14 | Transaction Cashback Amount | 23 | NUM | Cashback amount in major units |

| 15 | Original Amount (pre DCC) | 23 | NUM | DCC cardholder amount if applicable |

| 16 | Original Amount Currency (pre DCC) | 3 | CHAR | DCC cardholder currency |

| 17 | Transaction Date/Time | 15 | DATE | Format: YYYYMMDD HHMISS |

| 18 | Pricing Segment Code | 5 | CHAR | 5 character pricing segment code |

| 19 | Payee Reference (OTR) | 34 | CHAR | Originators Transaction Reference |

| 20 | Worldpay Transaction Source | 1 | CHAR | Source of original transaction |

| 21 | Airline ticket number | 255 | CHAR | Airline ticket number if applicable |

| 22 | Authorisation Method | 1 | NUM | Authorisation method |

| 23 | Authorisation Code | 9 | NUM | Authorisation code |

| 24 | Rejection Reason Code | 100 | VARCHAR | The rejection reason code |

| 25 | Trading Day | 8 | DATE | Transaction date in DDMMYY format |



\### Record Type 99 - Trailer Record



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | 99 - Trailer record |

| 2 | Record Count | 23 | NUM | Total records excluding last record |



---



\## 313 Funding Reconciliation File Structure



\### Record Type 00 - Header Record



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Header record, value '00' |

| 2 | Party Name | 80 | CHAR | Name of settlement party in data file |

| 3-6 | Reserved | - | - | Reserved fields |

| 7 | Settlement currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 12 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing Date | 8 | DATE | Data file processing date (YYYYMMDD) |



\### Record Type 01 - Today's Final Trading Amount



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Today's final trading amount, value '01' |

| 2 | Today's position | 50 | CHAR | Final funding position after deductions |

| 3-6 | Reserved | - | - | Reserved fields |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Amount | 23 | NUM | Final position (positive/negative) |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing Date | 8 | DATE | Data file processing date (YYYYMMDD) |



\### Record Type 02 - Balancing Items Which Affect Settlement



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Balancing Items, value '02' |

| 2 | Balance carried forward | 50 | CHAR | Statement of balancing items |

| 3 | Reserved | - | - | Reserved field |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5-6 | Reserved | - | - | Reserved fields |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Amount | 23 | NUM | Balancing item amount |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Due date | 8 | DATE | Payment due date (YYYYMMDD) |

| 14 | Processing date | 8 | DATE | Data file processing date (YYYYMMDD) |



\### Record Type 03 - Final Settlement Amount



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Final Settlement Amount, value '03' |

| 2 | Settlement amount | 50 | CHAR | Total Funding Settlement Amount |

| 3-6 | Reserved | - | - | Reserved fields |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Amount | 23 | NUM | Settlement amount value |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement Party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing Date | 8 | DATE | Data file processing date (YYYYMMDD) |



\### Record Type 04 - Individual Entries and Statement Narratives



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Individual entries, value '04' |

| 2 | Description | 50 | CHAR | Statement Narrative of Banking Entry |

| 3 | Reserved | - | - | Reserved field |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5-6 | Reserved | - | - | Reserved fields |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Amount | 23 | NUM | Entry amount value |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Due date | 8 | DATE | Payment due date |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 05 - Total Funding of FIS Acquired Transactions



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total funding FIS acquired, value '05' |

| 2 | Acquired Cards Subtotal | 50 | CHAR | Funding of FIS acquired transactions |

| 3 | Funding Bill ID | 25 | NUM | Unique funding batch reference |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5 | Reserved | - | - | Reserved field |

| 6 | Number of transactions | 10 | NUM | Funding count of acquired transactions |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value of Transactions | 15 | NUM | Funding value of transactions |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 12 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 06 - Total Funding of FIS Processed Transactions



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total Funding FIS Processed, value '06' |

| 2 | Processed Cards Subtotal | 52 | CHAR | Funding of FIS processed transactions |

| 3 | Bill ID | 25 | NUM | Unique Funding Batch reference |

| 4-5 | Reserved | - | - | Reserved fields |

| 6 | Number of transactions | 10 | NUM | Funding Count of transactions |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value of Transactions | 23 | NUM | Funding value of transactions |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing Date | 8 | DATE | Data file processing date |



\### Record Type 07 - Total Funding Adjustments



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total funding adjustments, value '07' |

| 2 | Funding adjustments | 44 | CHAR | Adjustments against funding amount |

| 3 | Bill ID | 25 | NUM | Unique funding batch reference |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5-6 | Reserved | - | - | Reserved fields |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value | 23 | NUM | Adjustment amount |

| 9 | From adjustment party | 15 | NUM | Party ID funds offset from |

| 10 | To adjustment party | 15 | NUM | Party ID funds offset to |

| 11 | Adjustment narrative | 18 | CHAR | Entered adjustment narrative |

| 12 | Party ID | 15 | CHAR | Settlement Party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 08 - Total FIS Acquired Transaction Charges



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record type | 2 | CHAR | Total FIS acquired charges, value '08' |

| 2 | Acquired charges Subtotal | 43 | CHAR | Charging of FIS acquired transactions |

| 3 | Bill ID | 25 | NUM | Invoice identifier for charging item |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5 | Reserved | - | - | Reserved field |

| 6 | Number of transactions | 10 | NUM | Charging count of transactions |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value of transactions | 23 | NUM | Charging value of transactions |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | VARCHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 09 - Total FIS Processed Transaction Charges



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record type | 2 | CHAR | Total FIS processed charges, value '09' |

| 2 | Processed charges subtotal | 43 | CHAR | Charging of FIS processed transactions |

| 3 | Bill ID | 25 | NUM | Invoice identifier for charging item |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5 | Reserved | - | - | Reserved field |

| 6 | Number of transactions | 10 | NUM | Charging Count of transactions |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value of transactions | 23 | NUM | Charging value of transactions |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 10 - Total Premium Cards Charges



| Field | Name | Length | Format | Description |

|-------|------|--------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total premium card changes, value '10' |

| 2 | Premium charges subtotal | 42 | CHAR | Charging of FIS Premium transactions |

| 3 | Bill ID | 25 | NUM | Invoice Identifier for charging item |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5 | Reserved | - | - | Reserved field |

| 6 | Number of transactions | 10 | NUM | Charging count of transactions |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value of transactions | 23 | NUM | Charging value of transactions |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 11 - Total Chargeback Processing Fees



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total chargeback processing fees, value '11' |

| 2 | Chargeback process subtotal | 42 | CHAR | Charging of FIS chargeback processing |

| 3 | Bill ID | 25 | NUM | Invoice Identifier for charging item |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5 | Reserved | - | - | Reserved field |

| 6 | Number of transactions | 10 | NUM | Charging count of transactions |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value of Transactions | 23 | NUM | Charging value of transactions |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing Date | 8 | DATE | Data file processing date |



\### Record Type 12 - Total Miscellaneous Charges



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total miscellaneous changes, value '12' |

| 2 | Miscellaneous charges | 42 | CHAR | Charging of FIS chargeback processing |

| 3 | Bill ID | 25 | NUM | Invoice Identifier for charging item |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5 | Reserved | - | - | Reserved field |

| 6 | Number of transactions | 10 | NUM | Charging count of transactions |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value of transactions | 23 | NUM | Charging value of transactions |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 13 - Total Charging Adjustments



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total charging adjustments, value '13' |

| 2 | Charging adjustments | 45 | CHAR | Adjustments against charging amount |

| 3 | Bill ID | 25 | NUM | Invoice Identifier for charging item |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5-6 | Reserved | - | - | Reserved fields |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value of transactions | 23 | NUM | Adjustment amount |

| 9 | From adjustment party | 15 | NUM | Party ID funds offset from |

| 10 | To adjustment party | 15 | NUM | Party ID funds offset to |

| 11 | Adjustment narrative | 18 | CHAR | Entered adjustment narrative |

| 12 | Party ID | 15 | CHAR | Settlement Party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 14 - Total Tax Applied



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total tax applied, value '14' |

| 2 | TAX | 11 | CHAR | Tax applied |

| 3 | Reserved | - | - | Reserved field |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5-6 | Reserved | - | - | Reserved fields |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Tax applied | 23 | CHAR | Value of Tax charged |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 15 - Total Chargebacks Charged



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Total chargebacks charged, value '15' |

| 2 | Chargebacks subtotal | 17 | CHAR | Chargebacks Total |

| 3 | Reserved | - | - | Reserved field |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5 | Reserved | - | - | Reserved field |

| 6 | Number of chargebacks | 10 | NUM | Count of chargebacks received |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value | 23 | NUM | Value of chargebacks charged |

| 9-11 | Reserved | - | - | Reserved fields |

| 12 | Party ID | 15 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | DATE | Data file processing date |



\### Record Type 16 - Chargeback Details



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Chargeback details value '16' |

| 2 | Pricing segment name | 50 | CHAR | Pricing Segment Name |

| 3 | Pricing segment code | 5 | CHAR | 5-character pricing segment code |

| 4 | Masked PAN | 19 | NUM | Masked PAN (actual or tokenised) |

| 5 | Reserved | - | - | Reserved field |

| 6 | Dispute ID | 20 | NUM | Unique reference for dispute |

| 7 | Currency | 3 | CHAR | Chargeback currency |

| 8 | Chargeback value | 23 | NUM | Value of chargebacks charged |

| 9 | ARN | 23 | VARCHAR | Acquirer reference number |

| 10 | Airline ticket | 255 | CHAR | Airline ticket number if applicable |

| 11 | Payee Reference (OTR) | 34 | CHAR | Originators Transaction Reference |

| 12 | Party ID | 12 | CHAR | Settlement party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing date | 8 | NUM | Data file processing date |



\### Record Type 17 - Chargeback Adjustments



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | CHAR | Chargeback adjustments, value '17' |

| 2 | Chargeback adjustment | TBC | CHAR | Adjustments against chargebacks |

| 3 | Reserved | - | - | Reserved field |

| 4 | Payment instruction ID | 20 | VARCHAR | Unique payment reference |

| 5-6 | Reserved | - | - | Reserved fields |

| 7 | Currency | 3 | CHAR | Currency in ISO Alpha standards |

| 8 | Value | 23 | NUM | Adjustment amount |

| 9 | From adjustment party | 15 | CHAR | Party ID funds charged from |

| 10 | To adjustment party | 15 | CHAR | Party ID funds charged to |

| 11 | Adjustment narrative | 18 | CHAR | Entered adjustment narrative |

| 12 | Party ID | 15 | CHAR | Party identifier |

| 13 | Reserved | - | - | Reserved field |

| 14 | Processing Date | 8 | DATE | Data file processing date |



\### Record Type 99 - Trailer Record



| Field | Name | Max Length | Format | Description |

|-------|------|------------|--------|-------------|

| 1 | Record Type | 2 | NUM | Trailer record, value = '99' |

| 2 | Record count | 10 | NUM | Total records excluding last record |



---



\## Code Mappings and Reference Data



\### Transaction Types (312 Record Types 01/02/03, Field 7)

\- Purchase

\- Purchase with Cashback

\- Cash advance

\- Quasi cash

\- Fee collection

\- Refund

\- Originating Credit Transfer (OCT)

\- Fee disbursement

\- Account enquiry



\### Transaction Sources (312 Record Types 01/02/03, Field 26/20/21)



| Code | Description |

|------|-------------|

| 0 | Offline key entry |

| 1 | Mail order - telephone |

| 2 | ICC and online PIN |

| 3 | ICC and offline PIN |

| 4 | Signed voucher - magnetic stripe captured |

| 5 | Signed voucher keyed at POS |

| 6 | Unattended device without PIN |

| 7 | Pin verified transaction recovered after sale |

| 8 | Terminal recovery keyed by acceptor |

| 9 | Terminal recovery keyed by acquirer |

| A | ICC and signature (or no cardholder verification) |

| B | ICC fallback transaction to magnetic stripe |

| C | Secure transaction with cardholder certificate |

| D | Non Authenticated Security transaction with 3D Secure/SPA UCAF merchant certificate |

| E | Non Authenticated transaction with no 3D Secure/SPA UCAF merchant certificate eg SSL protocol |

| F | E-commerce static cardholder authentication |

| M | qVSDC or M/Chip contactless transactions |

| N | MSD Contactless transactions |



\### Authorisation Methods (312 Record Types 01/02/03, Field 28/23/22)



| Code | Description |

|------|-------------|

| 0 | On line to Acquirer |

| 1 | Voice to Acquirer |

| 2 | Terminal |

| 3 | Voice Auth (4) |

| 4 | Voice Auth (5) |

| 5 | Merchant provided value |

| 6 | Authorised unknown |



\### Acquired/Processing Flags (312 Record Type 01, Field 18)



| Code | Description |

|------|-------------|

| A | FIS Acquired Transaction |

| P | FIS processed Transaction |



\### Cleared/Assessment Flags (312 Record Type 02, Field 9)



| Code | Description |

|------|-------------|

| C | Transaction successfully cleared |

| R | Transaction under review |



\### Pricing Segment Codes



| Code | Description |

|------|-------------|

| AC000 | MasterCard Cr Per |

| ACMCW | MasterCard Signia |

| ACMCY | MasterCardDr Per Int |

| ACMNW | MasterCard World |

| AS000 | All Star |

| AX000 | American Express |

| BC000 | Visa Credit Personal |

| BCVIY | Visa Dr Per Int |

| DC000 | Diners Club/Discover |

| DE000 | Visa Debit Per |

| DECOM | Visa Debit Com |

| DM000 | Dr MasterCard EEA |

| DMCOM | Debit MasterCard Com |

| DMMCY | Debit MasterCard Per Int |

| JC000 | JCB |

| KF000 | Keyfuels |

| LS000 | LaSer |

| OD000 | Overdrive |

| PE000 | Visa Electron |

| PECRE | Visa Electron Credit |

| PEVIY | Visa Elec Per Int |

| PMCOM | Maestro Com |

| PMDOM | Maestro Per |

| PMINC | Maestro Intl Com |

| PMINP | Maestro Intl Per |

| SC000 | Supercharge |

| SE000 | Sears |

| VP002 | MasterCard Comm |

| VPMCB | MasterCard Business |

| VPMCF | MasterCard Fleet |

| VPMCO | MasterCard Corporate |

| VPMCP | MasterCard Purchase |

| VPMCX | MasterCardDr Com Int |

| VPVIB | Visa Business |

| VPVID | Visa Commerce |

| VPVIR | Visa Corporate |

| VPVIS | Visa Purchasing |

| VPVIX | Visa Dr Com Intl |



\### Rejection Reason Codes (312 Record Type 03, Field 24)



| Code | Reason |

|------|--------|

| 7610111943 | Transaction from Merchant Marked as do not settle |

| 7400110013 | Txn exceeded refund/reversal amount limit |

| 7400110083 | Refund/rev velocity check failed |

| 7400310013 | Refund amt total amt |

| 7400310023 | Refund amt total amt trans value purchase to rfd |

| 7400310033 | Refund amt total amt |

| 7400310043 | Refund amt trans value |

| 7400310073 | Refund amt total amt |

| 7400310083 | Refund amt total amt |

| 1300899993 | JCB txn marked as invalid 2019.3 release |

| 1630120003 | BDR Drop Reason |

| 1630110453 | Merchant Does Not Accept Commercial Card |

| 1630110943 | Route Scheme data is missing |

| 1610110803 | Commercial txn amt doesn't equal parts minus dis |

| 1610110823 | Commercial txn without commodity code |

| 1610110833 | Commercial txn commodity code value is invalid |

| 1610110863 | Commercial transaction must have invoice |

| 1610113023 | Card Acceptor Business Code (MCC) |

| 1610111913 | POS Entry Mode Value is Not Supported |

| 1610111533 | Invalid Authorisation Code |

| 1610111473 | Original Transaction with Lifecycled Merchant |

| 1610111013 | No Cash Amount for PWCB Transaction |

| 1610111393 | Cashback Amount exceeds merchant limit |

| 1610111773 | Cashback amt must not be present for non PWCB |

| 1610111263 | Expiry Date Exceeds Scheme Validity Period |

| 1610111253 | Invalid Expiry Date |

| 1610111203 | Transaction Type Not Permissible for Merchant |

| 1610111153 | Original Credit Transaction Amt Non Wire Transfer |

| 1610110033 | Amount Zero |

| 1610110663 | Crdhlder No - Inv Chck Digit |

| 1610110513 | Trans Type Not Pemissible For Scheme |

| 1610110253 | Invalid Auth Code |

| 1610110283 | Invalid Auth Code |

| 1610110223 | Invalid Transaction Time |

| 1610110363 | Cashback Amount Exceeds Country Limit |

| 1610110173 | Transaction Date is Greater then CPD |

| 1610110123 | Length of Pan invalid |

| 1610113093 | Authorisation Code is invalid for Mastercard Txn |

| 1300827893 | Duplicate Transaction |

| 1300727893 | The payment/reversal is a duplicate |

| 1300227893 | The payment/reversal is a duplicate |

| 1306627893 | Duplicate Transaction |

| 1610110143 | Amount Exceeds Scheme Product Limit |

| 1300599993 | DINERS txn marked as invalid 2019.4 release |

| 1610111243 | Zero Expiry Date Not Valid For Non Visa,MC and Amex |



\### Balancing Items (313 Record Type 02, Field 2)



\- Balance Carried Forward

\- Deferred Payments previously Held

\- Deferred Payments

\- Released Payments Previously Deferred

\- Released Payments Previously Held

\- Balance Held Net Credit - Below Threshold

\- Balance Held Net Credit - Bank Details Incorrect

\- Balance Held Net Credit - Credits Suspended

\- Balance Held Net Credit - Non-Banking Day

\- Balance Held Net Debit

\- Direct Debit Deferred

\- Direct Debit Released



\### Funding Adjustments (313 Record Type 07, Field 2)



\- Funds Withheld

\- Offset Adjustments

\- Funding Adjustment



---



\## Abbreviations and Terms



| Abbreviation | Definition |

|--------------|------------|

| \*\*DART\*\* | Data Analytics Reconciliation Tool |

| \*\*FIS\*\* | Financial services company (payment processor) |

| \*\*CSV\*\* | Comma Separated Values |

| \*\*PAN\*\* | Primary Account Number (card number) |

| \*\*BIN\*\* | Bank Identification Number |

| \*\*MCC\*\* | Merchant Classification Code |

| \*\*OCT\*\* | Original Credit Transfer |

| \*\*DCC\*\* | Dynamic Currency Conversion |

| \*\*OTR\*\* | Originators Transaction Reference |

| \*\*ARN\*\* | Acquirer Reference Number |

| \*\*ISO\*\* | International Organization for Standardization |

| \*\*MOTO\*\* | Mail Order/Telephone Order |

| \*\*eCom\*\* | Electronic Commerce |

| \*\*PWCB\*\* | Purchase With Cashback |

| \*\*3D Secure\*\* | Three-Domain Secure (authentication protocol) |

| \*\*SPA\*\* | Secure Payment Application |

| \*\*UCAF\*\* | Universal Cardholder Authentication Field |

| \*\*SSL\*\* | Secure Sockets Layer |

| \*\*qVSDC\*\* | Quick Visa Smart Debit/Credit |

| \*\*M/Chip\*\* | Mastercard chip technology |

| \*\*MSD\*\* | Magnetic Stripe Data |

| \*\*ICC\*\* | Integrated Circuit Card (chip card) |

| \*\*PIN\*\* | Personal Identification Number |

| \*\*POS\*\* | Point of Sale |

| \*\*EEA\*\* | European Economic Area |

| \*\*JCB\*\* | Japan Credit Bureau |

| \*\*UTC\*\* | Coordinated Universal Time |



\## Data Formats



| Format | Description | Example |

|--------|-------------|---------|

| \*\*DATE\*\* | Date format YYYYMMDD | 20240315 |

| \*\*NUM\*\* | Numeric value | 12345 |

| \*\*CHAR\*\* | Character string | ABC123 |

| \*\*VARCHAR\*\* | Variable character string | Variable length text |



\## Currency Codes



All currency fields use \*\*ISO Alpha-3 standards\*\*:

\- USD (US Dollar)

\- EUR (Euro)

\- GBP (British Pound)

\- etc.



\## Amount Fields



\- All monetary amounts are in \*\*major units\*\* with exponent applied

\- Amounts stated to \*\*6 decimal points\*\* with trailing zeros dropped

\- Can be \*\*positive or negative\*\* values

\- Field lengths typically \*\*23 characters\*\* for transaction amounts



---



\## Contact Information



\### Scheme Compliance \& Management

\- \*\*Address\*\*: Level 5, Walbrook Building, 25 Walbrook, London, EC4N 8AF

\- \*\*Email\*\*: SC\&Mqueries@Worldpay.com



\### Payment Solutions Enablement (Technical Queries)

\- \*\*Address\*\*: Level 5, Walbrook Building, 25 Walbrook, London, EC4N 8AF

\- \*\*Email\*\*: ask-pse@Worldpay.com



---



\## Version Information



\- \*\*312 File Version\*\*: V3.4 (April 9, 2020)

\- \*\*313 File Version\*\*: V3.3 (April 9, 2020)

\- \*\*Document Classification\*\*: Confidential

\- \*\*Author\*\*: FIS Scheme Compliance and Management / Worldpay Scheme Compliance and Management

