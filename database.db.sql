BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "stock_data" (
	"transaction_id"	INTEGER,
	"ticker"	TEXT,
	"Stock"	TEXT,
	"Sector"	TEXT,
	"Price"	REAL,
	"Quantity"	INTEGER,
	PRIMARY KEY("transaction_id" AUTOINCREMENT)
);
INSERT INTO "stock_data" ("transaction_id","ticker","Stock","Sector","Price","Quantity") VALUES (0,'MCD','McDonald''s Corporation','Consumer Cyclical',275.540008544922,2),
 (1,'AAPL','Apple Inc.','Technology',150.25,50),
 (2,'GOOGL','Alphabet Inc. (Class A)','Technology',2850.75,30),
 (3,'AMZN','Amazon.com Inc.','Consumer Discretionary',3400.5,20),
 (4,'MSFT','Microsoft Corporation','Technology',310.8,40),
 (5,'JPM','JPMorgan Chase & Co.','Financials',155.6,35),
 (6,'V','Visa Inc.','Information Technology',250.15,12);
COMMIT;
