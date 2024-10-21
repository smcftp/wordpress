from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, BigInteger, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import os
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:DNzsRAQGZNwwXfOLSrnQCxuNFnVgotWK@autorack.proxy.rlwy.net:44307/railway")

# Настройка SQLAlchemy
engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Модель пользователя
class User(Base):
    __tablename__ = 'users'

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=False)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)

    sites = relationship('Site', back_populates='owner')
    
    def __repr__(self):
        return f"<User(id={self.id}, title={self.username}, url={self.email})>"

# Модель сайта
class Site(Base):
    __tablename__ = 'sites'

    id = Column(Integer, primary_key=True, index=True, autoincrement=False)
    user_id = Column(BigInteger, ForeignKey('users.id'))
    site_url = Column(String, index=True)
    site_name = Column(String)
    
    last_analysis_date = Column(DateTime, default=datetime.utcnow)  # Дата последнего анализа (по умолчанию текущее время)
    publication_interval = Column(Integer, default=183)  # Интервал публикации в часах (4 раза в месяц ~183 часа)


    owner = relationship('User', back_populates='sites')
    articles = relationship('Article', back_populates='site')
    
    def __repr__(self):
        return f"<Site(id={self.id}, site_url={self.site_url}, site_name={self.site_name}, last_analysis_date={self.last_analysis_date}, publication_interval={self.publication_interval})>"

# Модель статьи
class Article(Base):
    __tablename__ = 'articles'

    id = Column(Integer, primary_key=True, index=True, autoincrement=False)
    site_id = Column(Integer, ForeignKey('sites.id'))
    article_title = Column(String, index=True)
    article_url = Column(String)

    site = relationship('Site', back_populates='articles')
    
    def __repr__(self):
        return f"<Article(id={self.id}, title={self.article_title}, url={self.article_url})>"

# Создание всех таблиц
Base.metadata.create_all(bind=engine)


from sqlalchemy.orm import Session

def delete_articles_by_site_id(db: Session, site_id: int):
    try:
        # Удаление всех статей, связанных с конкретным сайтом
        db.query(Article).filter(Article.site_id == site_id).delete()
        
        # Применение изменений в базе данных
        db.commit()
        
        print(f"Все статьи, связанные с сайтом с id {site_id}, были успешно удалены.")
    except Exception as e:
        db.rollback()  # Откатить изменения в случае ошибки
        print(f"Ошибка при удалении статей: {e}")

# Пример использования
if __name__ == "__main__":
    # Создаем сессию
    db = SessionLocal()
    
    site_id = 13  # Укажите идентификатор сайта
    delete_articles_by_site_id(db, site_id)

    # Закрываем сессию
    db.close()
