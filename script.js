const todoForm = document.getElementById("todo-form");
const todoInput = document.getElementById("todo-input");
const memoInput = document.getElementById("memo-input");
const priorityInput = document.getElementById("priority-input");
const categoryInput = document.getElementById("category-input");
const dueDateInput = document.getElementById("due-date-input");
const reminderDaysInput = document.getElementById("reminder-days-input");
const repeatMonthlyInput = document.getElementById("repeat-monthly-input");
const repeatDayInput = document.getElementById("repeat-day-input");
const todoList = document.getElementById("todo-list");
const overdueList = document.getElementById("overdue-list");
const overdueSection = document.getElementById("overdue-section");
const overdueCount = document.getElementById("overdue-count");
const submitButton = document.getElementById("submit-button");
const cancelEditButton = document.getElementById("cancel-edit-button");
const prevMonthButton = document.getElementById("prev-month-button");
const nextMonthButton = document.getElementById("next-month-button");
const calendarTitle = document.getElementById("calendar-title");
const calendarGrid = document.getElementById("calendar-grid");
const selectedDateTitle = document.getElementById("selected-date-title");
const selectedDateList = document.getElementById("selected-date-list");
const calendarAddButton = document.getElementById("calendar-add-button");
const selectedDateAddButton = document.getElementById("selected-date-add-button");
const todayLabel = document.getElementById("today-label");
const remainingCount = document.getElementById("remaining-count");
const completedCount = document.getElementById("completed-count");
const progressText = document.getElementById("progress-text");
const progressPercent = document.getElementById("progress-percent");
const progressBar = document.getElementById("progress-bar");
const achievementPercent = document.getElementById("achievement-percent");
const highCount = document.getElementById("high-count");
const mediumCount = document.getElementById("medium-count");
const lowCount = document.getElementById("low-count");
const recordHighCount = document.getElementById("record-high-count");
const recordMediumCount = document.getElementById("record-medium-count");
const recordLowCount = document.getElementById("record-low-count");
const recordHighBar = document.getElementById("record-high-bar");
const recordMediumBar = document.getElementById("record-medium-bar");
const recordLowBar = document.getElementById("record-low-bar");
const formTitle = document.getElementById("form-title");
const navButtons = document.querySelectorAll(".nav-button");
const viewButtons = document.querySelectorAll("[data-view]");
const screens = document.querySelectorAll(".screen");

let todos = loadTodos();
let recurringTodos = loadRecurringTodos();
let editingIndex = null;
let calendarDate = getDateOnly(new Date());
let selectedDateText = "";
let lastAddedTodoId = "";
createRecurringTodosForMonth(calendarDate);
renderTodos();
renderCalendar();
renderStats();
setTodayLabel();

todoForm.addEventListener("submit", function (event) {
  event.preventDefault();

  const todoText = todoInput.value.trim();

  if (todoText === "") {
    return;
  }

  if (editingIndex === null && repeatMonthlyInput.checked) {
    addRecurringTodo(todoText, memoInput.value, priorityInput.value, categoryInput.value, repeatDayInput.value, reminderDaysInput.value);
  } else if (editingIndex === null) {
    addTodo(todoText, memoInput.value, priorityInput.value, categoryInput.value, dueDateInput.value, reminderDaysInput.value);
  } else {
    updateTodo(editingIndex, todoText, memoInput.value, priorityInput.value, categoryInput.value, dueDateInput.value, reminderDaysInput.value);
  }

  resetForm();
  showScreen("home");
});

cancelEditButton.addEventListener("click", function () {
  resetForm();
  showScreen("home");
});

calendarAddButton.addEventListener("click", function () {
  openFormForDate(selectedDateText);
});

selectedDateAddButton.addEventListener("click", function () {
  openFormForDate(selectedDateText);
});

viewButtons.forEach(function (button) {
  button.addEventListener("click", function () {
    showScreen(button.dataset.view);
  });
});

document.querySelectorAll("[data-priority-filter]").forEach(function (button) {
  button.addEventListener("click", function () {
    showScreen("home");
  });
});

prevMonthButton.addEventListener("click", function () {
  calendarDate = new Date(calendarDate.getFullYear(), calendarDate.getMonth() - 1, 1);
  createRecurringTodosForMonth(calendarDate);
  renderCalendar();
  renderTodos();
  renderStats();
});

nextMonthButton.addEventListener("click", function () {
  calendarDate = new Date(calendarDate.getFullYear(), calendarDate.getMonth() + 1, 1);
  createRecurringTodosForMonth(calendarDate);
  renderCalendar();
  renderTodos();
  renderStats();
});

function addTodo(text, memo, priority, category, dueDate, reminderDays) {
  const todoId = createTodoId();

  todos.push({
    id: todoId,
    text: text,
    memo: memo,
    priority: priority,
    category: category,
    completed: false,
    dueDate: dueDate,
    reminderDays: getReminderDays(reminderDays),
    recurringTemplateId: ""
  });

  lastAddedTodoId = todoId;
  saveTodos();
  renderTodos();
  renderCalendar();
  renderStats();
}

function addRecurringTodo(text, memo, priority, category, repeatDay, reminderDays) {
  const recurringTodo = {
    id: String(Date.now()),
    text: text,
    memo: memo,
    priority: priority,
    category: category,
    day: getRepeatDay(repeatDay),
    reminderDays: getReminderDays(reminderDays)
  };

  recurringTodos.push(recurringTodo);
  saveRecurringTodos();
  createRecurringTodosForMonth(calendarDate);
  saveTodos();
  renderTodos();
  renderCalendar();
  renderStats();
}

function updateTodo(index, text, memo, priority, category, dueDate, reminderDays) {
  todos[index].text = text;
  todos[index].memo = memo;
  todos[index].priority = priority;
  todos[index].category = category;
  todos[index].dueDate = dueDate;
  todos[index].reminderDays = getReminderDays(reminderDays);

  saveTodos();
  renderTodos();
  renderCalendar();
  renderStats();
}

function renderTodos() {
  todoList.innerHTML = "";
  overdueList.innerHTML = "";
  let overdueTotal = 0;

  getSortedTodos().forEach(function (item) {
    const listItem = createTodoItem(item.todo, item.index);

    if (!item.todo.completed && getDueStatus(item.todo) === "overdue") {
      overdueList.appendChild(listItem);
      overdueTotal++;
      return;
    }

    todoList.appendChild(listItem);
  });

  overdueCount.textContent = overdueTotal + "件";
  overdueSection.classList.toggle("hidden", overdueTotal === 0);
}

function createTodoItem(todo, index) {
  const listItem = document.createElement("li");
  listItem.className = "todo-item";
  const dueStatus = getDueStatus(todo);

  if (todo.id === lastAddedTodoId) {
    listItem.classList.add("newly-added");
    window.setTimeout(function () {
      lastAddedTodoId = "";
    }, 420);
  }

  if (todo.completed) {
    listItem.classList.add("completed");
  }

  if (!todo.completed && dueStatus === "due-soon") {
    listItem.classList.add("due-soon");
  }

  if (!todo.completed && dueStatus === "overdue") {
    listItem.classList.add("overdue");
  }

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = todo.completed;

  const content = document.createElement("div");
  content.className = "todo-content";

  const span = document.createElement("span");
  span.className = "todo-text";
  span.textContent = todo.text;

  content.appendChild(span);

  const prioritySpan = document.createElement("span");
  prioritySpan.className = "todo-priority " + todo.priority;
  prioritySpan.textContent = "重要度: " + getPriorityLabel(todo.priority);
  content.appendChild(prioritySpan);

  const categorySpan = document.createElement("span");
  categorySpan.className = "todo-category";
  categorySpan.textContent = "種類: " + getCategoryLabel(todo.category);
  content.appendChild(categorySpan);

  if (todo.dueDate) {
    const dateSpan = document.createElement("span");
    dateSpan.className = "todo-date";
    dateSpan.textContent = "期日: " + formatDate(todo.dueDate);
    content.appendChild(dateSpan);
  }

  if (todo.memo.trim() !== "") {
    const memoSpan = document.createElement("span");
    memoSpan.className = "todo-memo";
    memoSpan.textContent = todo.memo;
    content.appendChild(memoSpan);
  }

  if (todo.recurringTemplateId) {
    const repeatSpan = document.createElement("span");
    repeatSpan.className = "todo-repeat";
    repeatSpan.textContent = "毎月くり返し";
    content.appendChild(repeatSpan);
  }

  if (!todo.completed && dueStatus !== "none") {
    const alertSpan = document.createElement("span");
    alertSpan.className = "todo-alert";
    alertSpan.textContent = getAlertText(dueStatus);
    content.appendChild(alertSpan);
  }

  const actions = document.createElement("div");
  actions.className = "todo-actions";

  const editButton = document.createElement("button");
  editButton.type = "button";
  editButton.className = "edit-button";
  editButton.textContent = "編集";

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "delete-button";
  deleteButton.textContent = "削除";

  checkbox.addEventListener("change", function () {
    todos[index].completed = checkbox.checked;
    listItem.classList.toggle("completed", checkbox.checked);
    saveTodos();
    renderCalendar();
    renderStats();

    window.setTimeout(function () {
      renderTodos();
    }, 280);
  });

  editButton.addEventListener("click", function () {
    startEditing(index);
  });

  deleteButton.addEventListener("click", function () {
    listItem.classList.add("removing");

    window.setTimeout(function () {
      todos.splice(index, 1);
      resetForm();
      saveTodos();
      renderTodos();
      renderCalendar();
      renderStats();
    }, 260);
  });

  listItem.appendChild(checkbox);
  listItem.appendChild(content);
  actions.appendChild(editButton);
  actions.appendChild(deleteButton);
  listItem.appendChild(actions);
  return listItem;
}

function saveTodos() {
  localStorage.setItem("todos", JSON.stringify(todos));
}

function saveRecurringTodos() {
  localStorage.setItem("recurringTodos", JSON.stringify(recurringTodos));
}

function loadTodos() {
  const savedTodos = localStorage.getItem("todos");

  if (savedTodos === null) {
    return [];
  }

  return JSON.parse(savedTodos).map(function (todo) {
    return {
      id: todo.id || createTodoId(),
      text: todo.text,
      memo: todo.memo || "",
      priority: todo.priority || "medium",
      category: todo.category || "other",
      completed: todo.completed,
      dueDate: todo.dueDate || "",
      reminderDays: getReminderDays(todo.reminderDays),
      recurringTemplateId: todo.recurringTemplateId || ""
    };
  });
}

function loadRecurringTodos() {
  const savedRecurringTodos = localStorage.getItem("recurringTodos");

  if (savedRecurringTodos === null) {
    return [];
  }

  return JSON.parse(savedRecurringTodos).map(function (todo) {
    return {
      id: todo.id,
      text: todo.text,
      memo: todo.memo || "",
      priority: todo.priority || "medium",
      category: todo.category || "other",
      day: getRepeatDay(todo.day),
      reminderDays: getReminderDays(todo.reminderDays)
    };
  });
}

function getDueStatus(todo) {
  if (!todo.dueDate) {
    return "none";
  }

  const today = getDateOnly(new Date());
  const dueDate = getDateOnly(parseDate(todo.dueDate));
  const differenceTime = dueDate.getTime() - today.getTime();
  const differenceDays = Math.ceil(differenceTime / (1000 * 60 * 60 * 24));
  const reminderDays = getReminderDays(todo.reminderDays);

  if (differenceDays < 0) {
    return "overdue";
  }

  if (differenceDays <= reminderDays) {
    return "due-soon";
  }

  return "none";
}

function getDateOnly(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function parseDate(dateText) {
  const dateParts = dateText.split("-");
  const year = Number(dateParts[0]);
  const month = Number(dateParts[1]) - 1;
  const day = Number(dateParts[2]);

  return new Date(year, month, day);
}

function formatDate(dateText) {
  const date = parseDate(dateText);
  const year = date.getFullYear();
  const month = date.getMonth() + 1;
  const day = date.getDate();

  return year + "年" + month + "月" + day + "日";
}

function getAlertText(dueStatus) {
  if (dueStatus === "overdue") {
    return "期日を過ぎています";
  }

  return "期日が近づいています";
}

function getReminderDays(reminderDays) {
  const days = Number(reminderDays);

  if (Number.isNaN(days) || days < 0) {
    return 0;
  }

  return days;
}

function startEditing(index) {
  const todo = todos[index];

  editingIndex = index;
  todoInput.value = todo.text;
  memoInput.value = todo.memo;
  priorityInput.value = todo.priority;
  categoryInput.value = todo.category;
  dueDateInput.value = todo.dueDate;
  reminderDaysInput.value = todo.reminderDays;
  repeatMonthlyInput.checked = false;
  repeatDayInput.value = "1";
  submitButton.textContent = "更新する";
  formTitle.textContent = "タスクを編集";
  cancelEditButton.classList.remove("hidden");
  todoInput.focus();
  showScreen("form");
}

function resetForm() {
  editingIndex = null;
  todoInput.value = "";
  memoInput.value = "";
  priorityInput.value = "medium";
  categoryInput.value = "other";
  dueDateInput.value = "";
  reminderDaysInput.value = "1";
  repeatMonthlyInput.checked = false;
  repeatDayInput.value = "1";
  submitButton.textContent = "追加する";
  formTitle.textContent = "新しいタスク";
  cancelEditButton.classList.add("hidden");
  todoInput.focus();
}

function showScreen(viewName) {
  screens.forEach(function (screen) {
    screen.classList.toggle("active", screen.id === viewName + "-screen");
  });

  navButtons.forEach(function (button) {
    button.classList.toggle("active", button.dataset.view === viewName);
  });
}

function openFormForDate(dateText) {
  resetForm();

  if (dateText !== "") {
    dueDateInput.value = dateText;
  }

  showScreen("form");
}

function renderStats() {
  const total = todos.length;
  const completed = todos.filter(function (todo) {
    return todo.completed;
  }).length;
  const remaining = total - completed;
  const percent = total === 0 ? 0 : Math.round((completed / total) * 100);
  const priorityCounts = {
    high: getTodosByPriority("high").length,
    medium: getTodosByPriority("medium").length,
    low: getTodosByPriority("low").length
  };
  const maxPriorityCount = Math.max(priorityCounts.high, priorityCounts.medium, priorityCounts.low, 1);

  remainingCount.textContent = remaining;
  completedCount.textContent = completed;
  progressText.textContent = "完了 " + completed + " / " + total;
  progressPercent.textContent = percent + "%";
  progressBar.style.width = percent + "%";
  achievementPercent.textContent = percent + "%";

  highCount.textContent = priorityCounts.high;
  mediumCount.textContent = priorityCounts.medium;
  lowCount.textContent = priorityCounts.low;

  recordHighCount.textContent = priorityCounts.high;
  recordMediumCount.textContent = priorityCounts.medium;
  recordLowCount.textContent = priorityCounts.low;
  recordHighBar.style.width = (priorityCounts.high / maxPriorityCount) * 100 + "%";
  recordMediumBar.style.width = (priorityCounts.medium / maxPriorityCount) * 100 + "%";
  recordLowBar.style.width = (priorityCounts.low / maxPriorityCount) * 100 + "%";
}

function setTodayLabel() {
  const today = new Date();
  const weekdays = ["日", "月", "火", "水", "木", "金", "土"];
  todayLabel.textContent = today.getMonth() + 1 + "月" + today.getDate() + "日 " + weekdays[today.getDay()] + "曜日";
}

function renderCalendar() {
  calendarGrid.innerHTML = "";
  calendarTitle.textContent = calendarDate.getFullYear() + "年" + (calendarDate.getMonth() + 1) + "月";

  const year = calendarDate.getFullYear();
  const month = calendarDate.getMonth();
  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);
  const todayText = formatDateValue(new Date());

  for (let i = 0; i < firstDay.getDay(); i++) {
    const emptyDay = document.createElement("div");
    emptyDay.className = "calendar-day empty";
    calendarGrid.appendChild(emptyDay);
  }

  for (let day = 1; day <= lastDay.getDate(); day++) {
    const date = new Date(year, month, day);
    const dateText = formatDateValue(date);
    const todosForDate = getTodosByDate(dateText);
    const dayButton = document.createElement("button");
    dayButton.type = "button";
    dayButton.className = "calendar-day";

    if (dateText === todayText) {
      dayButton.classList.add("today");
    }

    if (dateText === selectedDateText) {
      dayButton.classList.add("selected");
    }

    if (todosForDate.length > 0) {
      dayButton.classList.add("has-todos");
      dayButton.classList.add("priority-" + getHighestPriority(todosForDate));
    }

    const dayNumber = document.createElement("span");
    dayNumber.className = "calendar-day-number";
    dayNumber.textContent = day;
    dayButton.appendChild(dayNumber);

    if (todosForDate.length > 0) {
      const todoCount = document.createElement("span");
      todoCount.className = "calendar-todo-count";
      todoCount.textContent = todosForDate.length + "件";
      dayButton.appendChild(todoCount);
    }

    dayButton.addEventListener("click", function () {
      selectedDateText = dateText;
      renderCalendar();
      renderSelectedDateTodos();
    });

    calendarGrid.appendChild(dayButton);
  }

  renderSelectedDateTodos();
}

function renderSelectedDateTodos() {
  selectedDateList.innerHTML = "";

  if (selectedDateText === "") {
    selectedDateTitle.textContent = "日付を選択してください";
    selectedDateAddButton.classList.add("hidden");
    return;
  }

  selectedDateTitle.textContent = formatDate(selectedDateText) + "のTodo";
  selectedDateAddButton.classList.remove("hidden");
  const todosForDate = getTodosByDate(selectedDateText);

  if (todosForDate.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.className = "selected-date-empty";
    emptyItem.textContent = "この日のTodoはありません";
    selectedDateList.appendChild(emptyItem);
    return;
  }

  todosForDate.forEach(function (todo) {
    const item = document.createElement("li");
    const todoInfo = document.createElement("span");
    const editButton = document.createElement("button");
    const todoIndex = todos.indexOf(todo);

    todoInfo.textContent = getCategoryLabel(todo.category) + " / 重要度" + getPriorityLabel(todo.priority) + ": " + todo.text;

    if (todo.completed) {
      todoInfo.textContent = todoInfo.textContent + "（完了済み）";
    }

    editButton.type = "button";
    editButton.className = "edit-button";
    editButton.textContent = "編集";
    editButton.addEventListener("click", function () {
      startEditing(todoIndex);
    });

    item.appendChild(todoInfo);
    item.appendChild(editButton);
    selectedDateList.appendChild(item);
  });
}

function getTodosByDate(dateText) {
  return getSortedTodos()
    .filter(function (item) {
      return item.todo.dueDate === dateText;
    })
    .map(function (item) {
      return item.todo;
    });
}

function createRecurringTodosForMonth(date) {
  const year = date.getFullYear();
  const month = date.getMonth();
  let hasNewTodo = false;

  recurringTodos.forEach(function (recurringTodo) {
    const lastDay = new Date(year, month + 1, 0).getDate();
    const day = Math.min(recurringTodo.day, lastDay);
    const dueDate = formatDateValue(new Date(year, month, day));
    const alreadyExists = todos.some(function (todo) {
      return todo.recurringTemplateId === recurringTodo.id && todo.dueDate === dueDate;
    });

    if (!alreadyExists) {
      const todoId = createTodoId();

      todos.push({
        id: todoId,
        text: recurringTodo.text,
        memo: recurringTodo.memo,
        priority: recurringTodo.priority,
        category: recurringTodo.category,
        completed: false,
        dueDate: dueDate,
        reminderDays: recurringTodo.reminderDays,
        recurringTemplateId: recurringTodo.id
      });
      lastAddedTodoId = todoId;
      hasNewTodo = true;
    }
  });

  if (hasNewTodo) {
    saveTodos();
  }
}

function getSortedTodos() {
  return todos
    .map(function (todo, index) {
      return {
        todo: todo,
        index: index
      };
    })
    .sort(function (firstItem, secondItem) {
      return getPriorityScore(secondItem.todo.priority) - getPriorityScore(firstItem.todo.priority);
    });
}

function getTodosByPriority(priority) {
  return todos.filter(function (todo) {
    return todo.priority === priority;
  });
}

function getHighestPriority(todosForDate) {
  const highestTodo = todosForDate.reduce(function (currentHighest, todo) {
    if (getPriorityScore(todo.priority) > getPriorityScore(currentHighest.priority)) {
      return todo;
    }

    return currentHighest;
  });

  return highestTodo.priority;
}

function getPriorityScore(priority) {
  if (priority === "high") {
    return 3;
  }

  if (priority === "medium") {
    return 2;
  }

  return 1;
}

function getPriorityLabel(priority) {
  if (priority === "high") {
    return "高";
  }

  if (priority === "low") {
    return "低";
  }

  return "中";
}

function getCategoryLabel(category) {
  const categoryLabels = {
    work: "仕事",
    payment: "支払い",
    life: "生活",
    private: "プライベート",
    study: "勉強",
    health: "健康",
    shopping: "買い物",
    cleaning: "掃除",
    cooking: "料理",
    family: "家族",
    friend: "友人",
    travel: "旅行",
    hobby: "趣味",
    exercise: "運動",
    finance: "お金",
    medical: "病院",
    document: "書類",
    event: "イベント",
    idea: "アイデア",
    other: "その他"
  };

  return categoryLabels[category] || "その他";
}

function createTodoId() {
  return String(Date.now()) + "-" + Math.random().toString(16).slice(2);
}

function getRepeatDay(day) {
  const repeatDay = Number(day);

  if (Number.isNaN(repeatDay) || repeatDay < 1) {
    return 1;
  }

  if (repeatDay > 31) {
    return 31;
  }

  return repeatDay;
}

function formatDateValue(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");

  return year + "-" + month + "-" + day;
}
