from sqlalchemy.orm import Session
from sqlalchemy import desc
from src.db.database import User, Site, Article, Base
from sqlalchemy import text
from datetime import datetime

# Функция для создания нового пользователя
async def create_user(db: Session, user_id: str, username: str, email: str, password_hash: str):
    db_user = User(id=user_id, username=username, email=email, password_hash=password_hash)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

async def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

# Функция для получения пользователя по username
async def get_user_by_username(db: Session, user_username: str) -> bool:
    user = db.query(User).filter(User.username == user_username).first()
    return True if user else False

# Функция для получения пользователя по username
async def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

# Функция для получения сайта по user_id и site_url
async def get_site_by_user_and_url(db: Session, user_id: int, site_url: str):
    return db.query(Site).filter(Site.user_id == user_id, Site.site_url == site_url).first()

# Функция для получения всех статей сайта по site_id
async def get_articles_by_site_id(db: Session, site_id: int):
    return db.query(Article).filter(Article.site_id == site_id).all()

# Функция для получения пос=иска пользователя по имейл или юсернем
async def get_user_by_email_or_username(db: Session, email: str, username: str):
    return db.query(User).filter(
        (User.email == email) | (User.username == username)
    ).first()

# Функция для получения первой статьи сайта по site_id и удаления её после получения
async def get_and_delete_first_article_by_site_id(db: Session, site_id: int):
    # Получаем первую статью для данного сайта
    first_article = db.query(Article).filter(Article.site_id == site_id).order_by(Article.id).first()

    if first_article:
        # Удаляем статью из базы данных
        db.delete(first_article)
        db.commit()  # Подтверждаем изменения в базе данных

    return first_article  # Возвращаем статью до удаления

# Функция для получения ID самого последнего пользователя в таблице User
async def get_last_user_id(db: Session):
    # Получаем последнего пользователя по id в порядке убывания
    last_user = db.query(User).order_by(desc(User.id)).first()
    
    if last_user:
        return last_user.id  
    else:
        return None  
    
# Функция для получения ID самого последнего сайта в таблице Site
async def get_last_site_id(db: Session):
    # Получаем последний сайт по id в порядке убывания
    last_site = db.query(Site).order_by(desc(Site.id)).first()
    
    if last_site:
        return last_site.id  # Возвращаем ID последнего сайта
    else:
        return 0  # Если сайтов нет, возвращаем None
    
# Функция для получения ID последней статьи в таблице Article
async def get_last_article_id(db: Session):
    # Получаем последнюю статью по id в порядке убывания
    last_article = db.query(Article).order_by(desc(Article.id)).first()
    
    if last_article:
        return last_article.id  
    else:
        return 0

# Функция для добавления сайта
async def create_site(db: Session, site_id: str, user_id: int, site_url: str, site_name: str, analysis_date: datetime = None, publication_interval: int = 183):
    # Если дата анализа не предоставлена, используем текущее время
    if analysis_date is None:
        analysis_date = datetime.utcnow()

    # Создаем объект сайта с параметрами
    db_site = Site(
        id=site_id, 
        user_id=user_id, 
        site_url=site_url, 
        site_name=site_name, 
        last_analysis_date=analysis_date,  # Дата анализа (по умолчанию текущее время)
        publication_interval=publication_interval  # Интервал публикации (по умолчанию 183 часа)
    )

    db.add(db_site)  # Добавляем объект в сессию
    db.commit()  # Подтверждаем изменения
    db.refresh(db_site)  # Обновляем объект

    return db_site  # Возвращаем созданный сайт

# Функция для добавления статьи
async def create_article(db: Session, article_id: str, site_id: int, article_title: str, article_url: str):
    db_article = Article(id=article_id, site_id=site_id, article_title=article_title, article_url=article_url)
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article

# Функция для получения всех сайтов пользователя
async def get_sites_by_user(db: Session, user_id: int):
    return db.query(Site).filter(Site.user_id == user_id).all()

# Функция для получения всех статей по сайту
async def get_articles_by_site(db: Session, site_id: int):
    return db.query(Article).filter(Article.site_id == site_id).all()

# Функция для полной очистки базы данных, удаляя все записи из всех таблиц.
async def clear_database(session: Session):
    # Отключение проверки внешних ключей (если используете SQLite)
    # Начать транзакцию
    session.begin()

    # Получение всех таблиц
    meta = Base.metadata
    for table in reversed(meta.sorted_tables):
        print(f'Очистка таблицы {table}')
        session.execute(table.delete())

    # Зафиксировать изменения
    session.commit()
