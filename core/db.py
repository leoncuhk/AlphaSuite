from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.model import Base, Exchange
from load_cfg import DATABASE_URL

# --- Global Database Setup ---
# Create the engine and session factory once when the module is imported.
# This is the standard and most efficient practice for database applications.
engine = create_engine(DATABASE_URL, echo=False)  # Set echo=True for SQL debugging
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def initialize_database_schema():
    """
    Initializes the database by creating all necessary tables and populating
    the 'exchange' table with static data. This function is idempotent and
    can be safely run multiple times.
    """
    # Create all tables defined in the models (if they don't exist)
    Base.metadata.create_all(bind=engine)
    # Populate the exchange table with predefined values (if they don't exist)
    _populate_exchange_table()

def get_db():
    """
    Provides a database session from the global session factory.
    Ensures the session is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def close_database():
    """
    Disposes of the connection pool. This is typically called at application shutdown.
    """
    engine.dispose()

def _populate_exchange_table():
    """
    Populates the exchange table with static data. This is an internal function
    called by initialize_database_schema. It checks if the table is empty
    before inserting data to ensure it only runs once.
    """
    db = SessionLocal()
    try:
        # If the table already has data, do nothing.
        if db.query(Exchange).first():
            return

        insert_scripts = [
            (10,'Americas','United States of America','us','NYSE','New York Stock Exchange',None,'09:30','16:00', 'America/New_York'),
            (11,'Americas','United States of America','us','NMS','NASDAQ Stock Market',None,'09:30','16:00', 'America/New_York'),
            (12,'Americas','United States of America','us','PNK','OTC Markets',None,'09:30','16:00', 'America/New_York'),
            (20,'Americas','Canada','ca','TOR','Toronto Stock Exchange','.TO','09:30','16:00', 'America/Toronto'),
            (21,'Americas','Canada','ca','VAN','TSX Venture Exchange','.V','09:30','16:00', 'America/Toronto'),
            (22,'Americas','Canada','ca','CSE0','Canadian Securities Exchange','.CN','09:30','16:00', 'America/Toronto'),
            (23,'Americas','Canada','ca','NEO','NEO Exchange','.NE','09:30','16:00', 'America/Toronto'),
            (30,'Americas','Brazil','br','BSP','Sao Paolo Stock Exchange','.SA','10:00','17:30', 'America/Sao_Paulo'),
            (40,'Americas','Chile','cl','SAN','Santiago Stock Exchange','.SN','09:30','17:00', 'America/Santiago'),
            (50,'Americas','Venezuela','ve','CCS','Caracas Stock Exchange','.CR','10:00','14:30', 'America/Caracas'),
            (60,'Americas','Argentina','ar','BUE','Buenos Aires Stock Exchange','.BA','11:00','17:00', 'America/Argentina/Buenos_Aires'),
            (70,'Americas','Mexico','mx','MEX','Mexico Stock Exchange','.MX','08:30','15:00', 'America/Mexico_City'),
            (80,'APAC','New Zealand','nz','NZX','New Zealand Stock Exchange','.NZ','10:00','16:45', 'Pacific/Auckland'),
            (90,'APAC','Australia','au','ASX','Australian Stock Exchange','.AX','10:00','16:00', 'Australia/Sydney'),
            (100,'APAC','Japan','jp','SPK','Sapporo Stock Exchange','.S','09:00','15:00', 'Asia/Tokyo'),
            (101,'APAC','Japan','jp','TYO','Tokyo Stock Exchange','.T','09:00','15:00', 'Asia/Tokyo'),
            (110,'APAC','South Korea','kr','KOQ','KOSDAQ','.KQ','09:00','15:30', 'Asia/Seoul'),
            (111,'APAC','South Korea','kr','KOR','Korea Stock Exchange','.KS','09:00','15:30', 'Asia/Seoul'),
            (120,'APAC','Taiwan','tw','TWN','Taiwan Stock Exchange','.TW','09:00','13:30', 'Asia/Taipei'),
            (121,'APAC','Taiwan','tw','TWO','Taiwan OTC Exchange','.TWO','09:00','13:30', 'Asia/Taipei'),
            (130,'APAC','Singapore','sg','SGX','Singapore Stock Exchange','.SI','09:00','17:00', 'Asia/Singapore'),
            (140,'APAC','Malaysia','my','KLS','Kuala Lumpur Stock Exchange','.KL','09:00','17:00', 'Asia/Kuala_Lumpur'),
            (150,'APAC','China','cn','SSE','Shanghai Stock Exchange','.SS','09:30','15:00', 'Asia/Shanghai'),
            (151,'APAC','China','cn','SZSE','Shenzhen Stock Exchange','.SZ','09:30','15:00', 'Asia/Shanghai'),
            (160,'APAC','Hong Kong','hk','HKG','Hong Kong Stock Exchange','.HK','09:30','16:00', 'Asia/Hong_Kong'),
            (170,'APAC','Indonesia','id','IDX','Indonesia Stock Exchange','.JK','09:00','16:00', 'Asia/Jakarta'),
            (180,'APAC','Thailand','th','BKK','Stock Exchange of Thailand','.BK','10:00','16:30', 'Asia/Bangkok'),
            (190,'APAC','India','in','BSE','Bombay Stock Exchange','.BO','09:15','15:30', 'Asia/Kolkata'),
            (191,'APAC','India','in','NSE','National Stock Exchange of India','.NS','09:15','15:30', 'Asia/Kolkata'),
            (200,'APAC','Sri Lanka','lk','CSE','Colombo Stock Exchange','.CM','09:30','15:30', 'Asia/Colombo'),
            (210,'EMEA','Qatar','qa','DOH','Qatar Stock Exchange','.QA','10:00','13:15', 'Asia/Qatar'),
            (220,'EMEA','South Africa','za','JNB','Johannesburg Stock Exchange','.JO','09:00','17:00', 'Africa/Johannesburg'),
            (230,'EMEA','Israel','il','TLV','Tel Aviv Stock Exchange','.TA','09:45','17:30', 'Asia/Jerusalem'),
            (240,'EMEA','Russia','ru','MCX','Moscow Exchange','.ME','09:50','18:40', 'Europe/Moscow'),
            (250,'EMEA','Saudi Arabia','sa','SAU','Saudi Stock Exchange','.SR','10:00','15:00', 'Asia/Riyadh'),
            (260,'EMEA','Turkey','tr','IST','Borsa Istanbul','.IS','09:40','18:00', 'Europe/Istanbul'),
            (270,'EMEA','Egypt','eg','EGX','Egyptian Stock Exchange','.CA','10:00','14:30', 'Africa/Cairo'),
            (280,'EMEA','Austria','at','VIE','Vienna Stock Exchange','.VI','09:00','17:30', 'Europe/Vienna'),
            (290,'EMEA','Germany','de','BER','Berlin Stock Exchange','.BE','08:00','20:00', 'Europe/Berlin'),
            (291,'EMEA','Germany','de','GER','Deutsche Boerse XETRA','.DE','09:00','17:30', 'Europe/Berlin'),
            (292,'EMEA','Germany','de','DUS','Dusseldorf Stock Exchange','.DU','08:00','20:00', 'Europe/Berlin'),
            (293,'EMEA','Germany','de','FRA','Frankfurt Stock Exchange','.F','08:00','20:00', 'Europe/Berlin'),
            (294,'EMEA','Germany','de','HAM','Hamburg Stock Exchange','.HM','08:00','20:00', 'Europe/Berlin'),
            (295,'EMEA','Germany','de','HAN','Hanover Stock Exchange','.HA','08:00','20:00', 'Europe/Berlin'),
            (296,'EMEA','Germany','de','MUN','Munich Stock Exchange','.MU','08:00','20:00', 'Europe/Berlin'),
            (297,'EMEA','Germany','de','STU','Stuttgart Stock Exchange','.SG','08:00','22:00', 'Europe/Berlin'),
            (300,'EMEA','Belgium','be','BRU','Euronext Brussels','.BR','09:00','17:30', 'Europe/Brussels'),
            (310,'EMEA','Czech Republic','cz','PRG','Prague Stock Exchange','.PR','09:00','16:00', 'Europe/Prague'),
            (320,'EMEA','Denmark','dk','CPH','Nasdaq OMX Copenhagen','.CO','09:00','17:00', 'Europe/Copenhagen'),
            (330,'EMEA','Estonia','ee','TAL','Nasdaq OMX Tallinn','.TL','10:00','16:00', 'Europe/Tallinn'),
            (340,'EMEA','Finland','fi','HEL','Nasdaq OMX Helsinki','.HE','10:00','18:30', 'Europe/Helsinki'),
            (350,'EMEA','France','fr','PAR','Euronext Paris','.PA','09:00','17:30', 'Europe/Paris'),
            (360,'EMEA','Hungary','hu','BUD','Budapest Stock Exchange','.BD','09:00','16:00', 'Europe/Budapest'),
            (370,'EMEA','Ireland','ie','ISE','Euronext Dublin','.IR','08:00','16:30', 'Europe/Dublin'),
            (380,'EMEA','Italy','it','ETL','EuroTLX','.TI','08:00','17:30', 'Europe/Rome'),
            (381,'EMEA','Italy','it','MIL','Italian Stock Exchange','.MI','08:00','17:30', 'Europe/Rome'),
            (390,'EMEA','Latvia','lv','RIG','Nasdaq OMX Riga','.RG','10:00','16:00', 'Europe/Riga'),
            (400,'EMEA','Lithuania','lt','VNO','Nasdaq OMX Vilnius','.VS','10:00','16:00', 'Europe/Vilnius'),
            (410,'EMEA','Netherlands','nl','AMS','Euronext Amsterdam','.AS','09:00','17:30', 'Europe/Amsterdam'),
            (420,'EMEA','Norway','no','OSL','Oslo Stock Exchange','.OL','09:00','16:30', 'Europe/Oslo'),
            (430,'EMEA','Portugal','pt','LIS','Euronext Lisbon','.LS','08:00','16:30', 'Europe/Lisbon'),
            (440,'EMEA','Spain','es','MCE','Madrid SE C.A.T.S.','.MC','09:00','17:30', 'Europe/Madrid'),
            (450,'EMEA','Sweden','se','STO','Nasdaq OMX Stockholm','.ST','09:00','17:30', 'Europe/Stockholm'),
            (460,'EMEA','Switzerland','ch','SWX','Swiss Exchange','.SW','08:00','17:30', 'Europe/Zurich'),
            (470,'EMEA','United Kingdom','gb','LON','London Stock Exchange','.L','08:00','16:30', 'Europe/London'),
            (471,'EMEA','United Kingdom','gb','IOB','International Order Book','.IL','08:00','16:30', 'Europe/London'),
            (480,'EMEA','Greece','gr','ATH','Athens Stock Exchange','.AT','10:00','17:20', 'Europe/Athens'),
            (490,'EMEA','Iceland','is','ICE','Nasdaq OMX Iceland','.IC','09:30','15:30', 'Atlantic/Reykjavik'),
        ]

        for id, continent, country, country_code, exchange_code, name, suffix, open_time, close_time, timezone in insert_scripts:
            exchange = Exchange(id=id, continent=continent, country=country, country_code=country_code, exchange_code=exchange_code, name=name, suffix=suffix, open_time=open_time, close_time=close_time, timezone=timezone)
            db.add(exchange)
        db.commit()

    except Exception as e:
        db.rollback()
        print(f"An error occurred while populating the exchange table: {e}")
    finally:
        db.close()
